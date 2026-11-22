package proxy

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"quickwork-openai/internal/accounts"
	"quickwork-openai/internal/config"
)

type Server struct {
	cfg      *config.Config
	accounts *accounts.Manager
	tokens   *APIKeyAuth
	chat     func(context.Context, *Session, ChatInput, EventFunc) (ChatResult, error)
}

func NewServer(cfg *config.Config, mgr *accounts.Manager) *Server {
	return &Server{
		cfg:      cfg,
		accounts: mgr,
		tokens:   NewAPIKeyAuth(cfg.APIKeys),
		chat:     NewUpstream(cfg).Chat,
	}
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", s.handleHealth)
	mux.HandleFunc("/v1/models", s.handleModels)
	mux.HandleFunc("/v1/chat/completions", s.handleChat)
	return s.withCORS(mux)
}

func (s *Server) withCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Headers", "Authorization, Content-Type, X-Api-Key, X-Thread-Id")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok", "service": "quickwork-openai"})
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeOpenAIError(w, http.StatusMethodNotAllowed, "method_not_allowed", "GET required")
		return
	}
	if !s.tokens.Allowed(r) {
		writeOpenAIError(w, http.StatusUnauthorized, "invalid_api_key", "invalid proxy API key")
		return
	}
	created := time.Now().Unix()
	data := make([]map[string]any, 0, len(s.cfg.Models))
	for _, id := range s.cfg.Models {
		data = append(data, map[string]any{
			"id":       id,
			"object":   "model",
			"created":  created,
			"owned_by": "quickwork",
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": data})
}

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeOpenAIError(w, http.StatusMethodNotAllowed, "method_not_allowed", "POST required")
		return
	}
	if !s.tokens.Allowed(r) {
		writeOpenAIError(w, http.StatusUnauthorized, "invalid_api_key", "invalid proxy API key")
		return
	}

	var req ChatRequest
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 8<<20))
	if err := dec.Decode(&req); err != nil {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", "invalid JSON body: "+err.Error())
		return
	}
	if req.N > 1 {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", "n > 1 is not supported")
		return
	}

	prompt, err := formatPrompt(req.Messages)
	if err != nil {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	systemPromptText := systemPrompt(req.Messages)
	model := s.mapModel(req.Model)
	allowedTools := requestedToolNames(&req)
	choiceMode, choiceName := parseToolChoice(req.ToolChoice)
	if choiceMode == "function" && !allowedTools[choiceName] {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", "tool_choice names an undeclared tool")
		return
	}
	parallelTools := true
	if req.ParallelToolCalls != nil {
		parallelTools = *req.ParallelToolCalls
	}
	bridgePrompt := buildToolBridgePrompt(req.Tools, req.Functions, choiceMode, choiceName, parallelTools)
	bridgeActive := bridgePrompt != ""
	if bridgeActive {
		// Smart can consult its own workspace when the client catalog appears
		// only in system_prompt. Carry the catalog with the actual turn.
		systemPromptText += "\n\nThe client tool catalog and its wire format appear at the start of this turn. Use that protocol to send tool calls to the client through this relay."
		prompt = bridgePrompt + "\n\n" + prompt
	}
	bridgeAllowed := allowedTools
	if choiceMode == "function" && choiceName != "" {
		bridgeAllowed = map[string]bool{choiceName: true}
	}

	ctx, cancel := context.WithTimeout(r.Context(), s.cfg.RequestTimeout.Duration())
	defer cancel()

	threadID := strings.TrimSpace(r.Header.Get("X-Thread-Id"))
	id := "chatcmpl-" + strings.ReplaceAll(uuid(), "-", "")
	created := time.Now().Unix()
	in := ChatInput{
		Prompt:       prompt,
		Model:        model,
		ThreadID:     threadID,
		Title:        titleFor(lastUserText(req.Messages)),
		SystemPrompt: systemPromptText,
	}

	flusher, ok := w.(http.Flusher)
	if req.Stream && !ok {
		writeOpenAIError(w, http.StatusInternalServerError, "streaming_unavailable", "response writer does not support flushing")
		return
	}
	started := false
	writeRaw := func(payload map[string]any) {
		body, _ := json.Marshal(payload)
		_, _ = fmt.Fprintf(w, "data: %s\n\n", body)
		flusher.Flush()
	}
	ensureStarted := func() {
		if started {
			return
		}
		started = true
		w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("X-Accel-Buffering", "no")
		if threadID != "" {
			w.Header().Set("X-Thread-Id", threadID)
		}
		w.WriteHeader(http.StatusOK)
		writeRaw(map[string]any{
			"id":      id,
			"object":  "chat.completion.chunk",
			"created": created,
			"model":   model,
			"choices": []map[string]any{{
				"index":         0,
				"delta":         map[string]any{"role": "assistant"},
				"finish_reason": nil,
			}},
		})
	}
	writePayload := func(payload map[string]any) {
		ensureStarted()
		writeRaw(payload)
	}
	writeChunk := func(delta map[string]any, finish any) {
		writePayload(map[string]any{
			"id":      id,
			"object":  "chat.completion.chunk",
			"created": created,
			"model":   model,
			"choices": []map[string]any{{
				"index":         0,
				"delta":         delta,
				"finish_reason": finish,
			}},
		})
	}
	writeStreamError := func(message string) {
		writePayload(map[string]any{"error": map[string]any{
			"message": message,
			"type":    "upstream_error",
			"code":    "upstream_error",
		}})
	}

	sawText := false
	var res ChatResult
	var sess *Session
	var lastErr error
	for range s.maxAttempts(ctx) {
		acct, err := s.accounts.Next(ctx)
		if err != nil {
			lastErr = err
			break
		}
		sess = SessionFromAccount(acct, s.cfg)
		if err := sess.validate(); err != nil {
			s.accounts.ReportFailure(ctx, acct.ID, err)
			lastErr = err
			continue
		}
		res, err = s.chat(ctx, sess, in, func(ev StreamEvent) error {
			if req.Stream && !bridgeActive && ev.Type == "text" && ev.Text != "" {
				sawText = true
				writeChunk(map[string]any{"content": ev.Text}, nil)
			}
			return nil
		})
		if err == nil {
			lastErr = nil
			break
		}
		lastErr = err
		s.accounts.ReportFailure(ctx, acct.ID, err)
		if started || ctx.Err() != nil {
			break
		}
		// Nothing reached the client yet; reset per-attempt state and try
		// the next account in rotation.
		sawText = false
	}
	if lastErr != nil {
		if !started {
			status, code := http.StatusBadGateway, "upstream_error"
			if errors.Is(lastErr, accounts.ErrNoAccounts) {
				status, code = http.StatusUnauthorized, "upstream_session_unavailable"
			}
			writeOpenAIError(w, status, code, lastErr.Error())
			return
		}
		writeStreamError(lastErr.Error())
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
		return
	}
	text, calls, err := resolveToolCalls(res, bridgeAllowed, choiceMode, choiceName, parallelTools)
	if errors.Is(err, errToolMarker) && ctx.Err() == nil {
		// Reformat only the rejected response. Replaying the user's full task
		// could repeat upstream actions or change the intended invocation.
		log.Printf("client tool call needs repair: %v", err)
		repair := ChatInput{
			Model: model, Title: "Format client tool call",
			SystemPrompt: systemPromptText + "\n\nReformat the supplied draft as client tool calls. Preserve the intended arguments. If a required value is missing and cannot be recovered, report failure. Do not execute the task.",
			Prompt:       bridgePrompt + "\n\nReturn the corrected tool call(s) for this rejected draft. Keep the original tool and all parameter values. Use the declared tool names and complete JSON object arguments.\n" + res.Text,
		}
		fixed, repairErr := s.chat(ctx, sess, repair, nil)
		if repairErr != nil {
			writeOpenAIError(w, http.StatusBadGateway, "upstream_error", repairErr.Error())
			return
		}
		res.Usage.add(fixed.Usage)
		text, calls, err = resolveToolCalls(fixed, bridgeAllowed, choiceMode, choiceName, parallelTools)
		if err == nil && len(calls) == 0 {
			err = fmt.Errorf("%w: repair produced no tool call", errToolMarker)
		}
	}
	if err != nil {
		writeOpenAIError(w, http.StatusBadGateway, "invalid_tool_call", err.Error())
		return
	}
	if !req.Stream {
		if strings.TrimSpace(text) == "" && len(calls) == 0 {
			writeOpenAIError(w, http.StatusBadGateway, "upstream_error", "upstream returned no text or tool calls")
			return
		}
		message := map[string]any{"role": "assistant"}
		if strings.TrimSpace(text) == "" {
			message["content"] = nil
		} else {
			message["content"] = text
		}
		if len(calls) > 0 {
			message["tool_calls"] = calls
		}
		finish := "stop"
		if len(calls) > 0 {
			finish = "tool_calls"
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"id":      id,
			"object":  "chat.completion",
			"created": created,
			"model":   model,
			"choices": []map[string]any{{
				"index":         0,
				"message":       message,
				"finish_reason": finish,
			}},
			"usage": usageMap(res.Usage),
		})
		return
	}
	finish := "stop"
	if bridgeActive {
		if strings.TrimSpace(text) != "" {
			sawText = true
			writeChunk(map[string]any{"content": text}, nil)
		}
		for i, call := range calls {
			writeChunk(map[string]any{"tool_calls": []map[string]any{{
				"index": i, "id": call.ID, "type": call.Type, "function": call.Function,
			}}}, nil)
			finish = "tool_calls"
		}
	}
	if !sawText && finish == "stop" {
		writeStreamError("upstream returned no text or tool calls")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
		return
	}
	writeChunk(map[string]any{}, finish)
	if req.StreamOptions != nil && req.StreamOptions.IncludeUsage {
		writePayload(map[string]any{
			"id":      id,
			"object":  "chat.completion.chunk",
			"created": created,
			"model":   model,
			"choices": []any{},
			"usage":   usageMap(res.Usage),
		})
	}
	_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func (s *Server) maxAttempts(ctx context.Context) int {
	n, err := s.accounts.EnabledCount(ctx)
	if err != nil || n < 1 {
		return 1
	}
	max := s.cfg.MaxAttempts
	if max < 1 {
		max = 1
	}
	if n < max {
		return n
	}
	return max
}

func (s *Server) mapModel(model string) string {
	m := strings.ToLower(strings.TrimSpace(model))
	m = strings.TrimPrefix(m, "quickwork/")
	m = strings.TrimPrefix(m, "qw-")
	switch m {
	case "smart", "fast", "balanced", "auto":
		return m
	case "":
		return s.cfg.DefaultModel
	default:
		log.Printf("unknown model %q, using %s", model, s.cfg.DefaultModel)
		return s.cfg.DefaultModel
	}
}

func requestedToolNames(req *ChatRequest) map[string]bool {
	names := make(map[string]bool)
	for _, t := range req.Tools {
		if name := strings.TrimSpace(t.Function.Name); name != "" {
			names[name] = true
		}
	}
	for _, f := range req.Functions {
		if name := strings.TrimSpace(f.Name); name != "" {
			names[name] = true
		}
	}
	return names
}

func parseToolChoice(v any) (mode, name string) {
	switch x := v.(type) {
	case string:
		return strings.ToLower(strings.TrimSpace(x)), ""
	case map[string]any:
		if fn, ok := x["function"].(map[string]any); ok {
			if n, ok := fn["name"].(string); ok {
				return "function", strings.TrimSpace(n)
			}
		}
		if n, ok := x["name"].(string); ok {
			return "function", strings.TrimSpace(n)
		}
		if t, ok := x["type"].(string); ok {
			return strings.ToLower(strings.TrimSpace(t)), ""
		}
	}
	return "auto", ""
}

func toolAllowed(tc ToolCallState, allowed map[string]bool, mode, name string) bool {
	if mode == "none" || len(allowed) == 0 {
		return false
	}
	if !allowed[tc.Name] {
		return false
	}
	if mode == "function" && name != "" && tc.Name != name {
		return false
	}
	return true
}

func filterToolCalls(calls []ToolCallState, allowed map[string]bool, mode, name string, parallel bool) []ToolCall {
	out := make([]ToolCall, 0, len(calls))
	for _, tc := range calls {
		if !toolAllowed(tc, allowed, mode, name) {
			continue
		}
		if !parallel && len(out) > 0 {
			break
		}
		id := strings.TrimSpace(tc.ID)
		if id == "" {
			id = "call_" + strings.ReplaceAll(uuid(), "-", "")
		}
		toolName := strings.TrimSpace(tc.Name)
		if toolName == "" {
			toolName = "unknown"
		}
		args := strings.TrimSpace(tc.Arguments)
		if args == "" {
			args = "{}"
		}
		out = append(out, ToolCall{
			ID:   id,
			Type: "function",
			Function: FunctionCall{
				Name:      toolName,
				Arguments: args,
			},
		})
	}
	return out
}

func usageMap(u Usage) map[string]any {
	completion := u.CompletionTokens
	total := u.TotalTokens
	if total == 0 {
		total = u.PromptTokens + u.CacheReadTokens + u.CacheWriteTokens + completion
	}
	prompt := total - completion
	if prompt < 0 {
		prompt = u.PromptTokens + u.CacheReadTokens + u.CacheWriteTokens
	}
	m := map[string]any{
		"prompt_tokens":     prompt,
		"completion_tokens": completion,
		"total_tokens":      total,
	}
	if u.CacheReadTokens > 0 {
		m["prompt_tokens_details"] = map[string]any{"cached_tokens": u.CacheReadTokens}
	}
	return m
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeOpenAIError(w http.ResponseWriter, status int, code, message string) {
	writeJSON(w, status, map[string]any{
		"error": map[string]any{
			"message": message,
			"type":    code,
			"code":    code,
		},
	})
}
