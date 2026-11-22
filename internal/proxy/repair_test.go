package proxy

import (
	"context"
	"fmt"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"quickwork-openai/internal/accounts"
	"quickwork-openai/internal/config"
	"quickwork-openai/internal/store"
)

func TestRepairBeforeSendingResponse(t *testing.T) {
	const broken = `<client_tool_call>{"name":"lookup","arguments":{"key":"item "A""}}</client_tool_call>`
	const valid = `<client_tool_call>{"name":"lookup","arguments":{"key":"item \"A\""}}</client_tool_call>`
	for _, stream := range []bool{true, false} {
		for _, draft := range []string{valid, broken, "could not recover"} {
			cfg := config.Default()
			db, err := store.Open(filepath.Join(t.TempDir(), "test.db"))
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { db.Close() })
			err = db.Save(t.Context(), &store.Account{Email: "test@example.com", Provider: "social", WSHost: "example.com", AccessToken: "test", ExpiresAt: time.Now().Add(time.Hour).Unix()})
			if err != nil {
				t.Fatal(err)
			}
			s := NewServer(&cfg, accounts.New(db, &cfg))
			w := httptest.NewRecorder()
			requests := 0
			s.chat = func(ctx context.Context, sess *Session, in ChatInput, onEvent EventFunc) (ChatResult, error) {
				requests++
				if requests == 1 {
					return ChatResult{Text: broken, Usage: Usage{PromptTokens: 7}}, nil
				}
				if requests != 2 || in.ThreadID != "" || !strings.Contains(in.Prompt, broken) || strings.Contains(in.Prompt, "original task") || !strings.Contains(in.SystemPrompt, "caller rules") || w.Body.Len() != 0 {
					t.Fatal("unexpected repair request or prematurely written response")
				}
				return ChatResult{Text: draft, Usage: Usage{PromptTokens: 5}}, nil
			}
			req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(fmt.Sprintf(`{"model":"smart","messages":[{"role":"system","content":"caller rules"},{"role":"user","content":"original task"}],"tools":[{"type":"function","function":{"name":"lookup"}}],"stream":%t,"stream_options":{"include_usage":true}}`, stream)))
			req.Header.Set("X-Thread-Id", "caller-thread")
			s.Handler().ServeHTTP(w, req)
			body := w.Body.String()
			if requests != 2 {
				t.Fatal("repair count", requests)
			}
			if draft == valid {
				if w.Code != 200 || !strings.Contains(body, `"finish_reason":"tool_calls"`) || !strings.Contains(body, `"prompt_tokens":12`) || strings.Count(body, `"name":"lookup"`) != 1 || strings.Contains(body, "client_tool_call") || strings.Contains(body, "[DONE]") != stream {
					t.Fatalf("invalid success response: %s", body)
				}
			} else if w.Code != 502 || !strings.Contains(body, "invalid_tool_call") || strings.Contains(body, "data: ") {
				t.Fatalf("invalid error response: %s", body)
			}
		}
	}
}

func TestCompleteJSONWithoutFinalMarker(t *testing.T) {
	const draft = `<client_tool_call>{"tool_calls":[{"type":"function","function":{"name":"lookup","arguments":"{\"key\":\"A\"}"}}]}`
	text, calls, err := resolveToolCalls(ChatResult{Text: draft}, map[string]bool{"lookup": true}, "auto", "", true)
	if err != nil || text != "" || len(calls) != 1 || calls[0].Function.Arguments != `{"key":"A"}` {
		t.Fatalf("%q %+v %v", text, calls, err)
	}
}
