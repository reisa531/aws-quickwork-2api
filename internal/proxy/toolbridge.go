package proxy

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// ToolCallReply is a client-side tool call parsed from the model's text.
type ToolCallReply struct {
	ID        string
	Name      string
	Arguments string
}

type toolDefinition struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

type toolCallEnvelope struct {
	Function *struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	} `json:"function"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

const (
	clientToolCallOpen  = "<client_tool_call>"
	clientToolCallClose = "</client_tool_call>"
)

func collectToolDefinitions(tools []Tool, functions []ToolFunction) []toolDefinition {
	defs := make([]toolDefinition, 0, len(tools)+len(functions))
	for _, t := range tools {
		f := t.Function
		if strings.TrimSpace(f.Name) == "" {
			continue
		}
		defs = append(defs, toolDefinition{Name: f.Name, Description: f.Description, Parameters: f.Parameters})
	}
	for _, f := range functions {
		if strings.TrimSpace(f.Name) == "" {
			continue
		}
		defs = append(defs, toolDefinition{Name: f.Name, Description: f.Description, Parameters: f.Parameters})
	}
	return defs
}

// buildToolBridgePrompt teaches the upstream model a small text protocol for
// tools that only the caller can execute. QuickWork does not accept arbitrary
// client tool schemas, so the proxy has to carry them in the turn.
func buildToolBridgePrompt(tools []Tool, functions []ToolFunction, mode, name string, parallel bool) string {
	if mode == "none" {
		return ""
	}
	defs := collectToolDefinitions(tools, functions)
	if len(defs) == 0 {
		return ""
	}
	body, err := json.Marshal(defs)
	if err != nil {
		return ""
	}

	var b strings.Builder
	b.WriteString("<client_tool_protocol>\n")
	b.WriteString("The caller exposes these client tools:\n")
	b.Write(body)
	b.WriteString("\nWhen a listed tool is needed, request it from the client using the following wire format. The proxy converts this marker into a native tool call for the client. The client runs the tool in its own environment. Use the client tool for its advertised task, including lookups of client data. Output one marker per invocation, with the exact tool name and a JSON object matching its parameters:\n")
	b.WriteString(`<client_tool_call>{"name":"TOOL_NAME","arguments":{...}}</client_tool_call>`)
	b.WriteString("\nUse valid JSON string escaping for commands (for example, write \\\\s for a literal backslash followed by s). End the turn after the marker(s) and wait for the client. If the transcript contains a tool result, use that result to continue. When the task is complete, answer normally.\n")
	if mode == "function" && name != "" {
		fmt.Fprintf(&b, "You must call the tool %q.\n", name)
	}
	if mode == "required" {
		b.WriteString("You must call at least one tool.\n")
	}
	if !parallel {
		b.WriteString("Call at most one tool per turn.\n")
	}
	b.WriteString("</client_tool_protocol>")
	return b.String()
}

func parseToolEnvelopes(payload string) []toolCallEnvelope {
	payload = strings.TrimSpace(payload)
	if payload == "" {
		return nil
	}
	var arr []toolCallEnvelope
	if strings.HasPrefix(payload, "[") && json.Unmarshal([]byte(payload), &arr) == nil {
		return arr
	}
	var wrapper struct {
		ToolCalls []toolCallEnvelope `json:"tool_calls"`
	}
	if json.Unmarshal([]byte(payload), &wrapper) == nil && len(wrapper.ToolCalls) > 0 {
		return wrapper.ToolCalls
	}
	var one toolCallEnvelope
	if json.Unmarshal([]byte(payload), &one) == nil {
		return []toolCallEnvelope{one}
	}
	return nil
}

func repliesFromEnvelopes(envelopes []toolCallEnvelope, allowed map[string]bool) []ToolCallReply {
	out := make([]ToolCallReply, 0, len(envelopes))
	for _, env := range envelopes {
		if env.Function != nil && env.Name == "" && env.Arguments == nil {
			env.Name, env.Arguments = env.Function.Name, env.Function.Arguments
		}
		name := strings.TrimSpace(env.Name)
		if name == "" || !allowed[name] {
			continue
		}
		args := argumentString(env.Arguments)
		if !strings.HasPrefix(strings.TrimSpace(args), "{") || !json.Valid([]byte(args)) {
			continue
		}
		out = append(out, ToolCallReply{
			ID:        "call_" + strings.ReplaceAll(uuid(), "-", ""),
			Name:      name,
			Arguments: args,
		})
	}
	return out
}

// Model-written commands sometimes contain literal line breaks or regex
// escapes such as \s. Quote only bytes that JSON cannot already decode.
func quoteInvalidStringEscapes(payload string) string {
	var out strings.Builder
	inString := false
	for i := 0; i < len(payload); i++ {
		c := payload[i]
		switch {
		case c == '"':
			inString = !inString
		case inString && c == '\\' && i+1 < len(payload):
			next := payload[i+1]
			out.WriteByte('\\')
			if strings.ContainsRune(`"\/bfnrtu`, rune(next)) {
				i++
				out.WriteByte(next)
				continue
			}
		case inString && c < 0x20:
			fmt.Fprintf(&out, `\u%04x`, c)
			continue
		}
		out.WriteByte(c)
	}
	return out.String()
}

var errToolMarker = errors.New("upstream returned an incomplete or invalid client tool call")

// Read one complete JSON value before looking for its closing marker. A
// command can itself contain the literal marker, for example in a file edit.
func markedToolCalls(text string, allowed map[string]bool) (string, []ToolCallReply, error) {
	var clean strings.Builder
	var replies []ToolCallReply
	for {
		before, rest, found := strings.Cut(text, clientToolCallOpen)
		clean.WriteString(before)
		if !found {
			break
		}
		trimmed := strings.TrimLeft(rest, " \t\r\n")
		if !strings.HasPrefix(trimmed, "{") && !strings.HasPrefix(trimmed, "[") {
			clean.WriteString(clientToolCallOpen)
			text = rest
			continue
		}
		rest = trimmed
		dec := json.NewDecoder(strings.NewReader(rest))
		var raw json.RawMessage
		var payload, after string
		if err := dec.Decode(&raw); err == nil {
			payload = string(raw)
			tail := strings.TrimLeft(rest[dec.InputOffset():], " \t\r\n")
			if tail == "" {
				// The JSON value is complete and only its final tag is missing.
				after, found = "", true
			} else {
				after, found = strings.CutPrefix(tail, clientToolCallClose)
			}
		} else {
			payload, after, found = strings.Cut(rest, clientToolCallClose)
			payload = quoteInvalidStringEscapes(payload)
		}
		if !found {
			return "", nil, fmt.Errorf("%w: missing closing marker", errToolMarker)
		}
		envelopes := parseToolEnvelopes(payload)
		if len(envelopes) == 0 {
			return "", nil, fmt.Errorf("%w: could not decode call JSON", errToolMarker)
		}
		calls := repliesFromEnvelopes(envelopes, allowed)
		if len(calls) == 0 || len(calls) != len(envelopes) {
			return "", nil, fmt.Errorf("%w: tool name or arguments do not match the client protocol", errToolMarker)
		}
		replies = append(replies, calls...)
		text = after
	}
	if len(replies) == 0 {
		return clean.String(), nil, nil
	}
	return strings.TrimSpace(clean.String()), replies, nil
}

// extractBareToolCall handles models that answer with a single JSON object
// instead of the requested marker.
func extractBareToolCall(text string, allowed map[string]bool) (string, []ToolCallReply) {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return text, nil
	}
	if strings.HasPrefix(trimmed, "```") {
		if i := strings.Index(trimmed, "\n"); i >= 0 {
			trimmed = strings.TrimSuffix(strings.TrimSpace(trimmed[i+1:]), "```")
			trimmed = strings.TrimSpace(trimmed)
		}
	}
	if !strings.HasPrefix(trimmed, "{") && !strings.HasPrefix(trimmed, "[") {
		return text, nil
	}
	replies := repliesFromEnvelopes(parseToolEnvelopes(trimmed), allowed)
	if len(replies) == 0 {
		return text, nil
	}
	return "", replies
}

func replyToolCalls(replies []ToolCallReply) []ToolCall {
	out := make([]ToolCall, 0, len(replies))
	for _, r := range replies {
		out = append(out, ToolCall{
			ID:   r.ID,
			Type: "function",
			Function: FunctionCall{
				Name:      r.Name,
				Arguments: r.Arguments,
			},
		})
	}
	return out
}

func resolveToolCalls(res ChatResult, allowed map[string]bool, mode, name string, parallel bool) (string, []ToolCall, error) {
	if mode == "none" || len(allowed) == 0 {
		return res.Text, nil, nil
	}
	text, replies, err := markedToolCalls(res.Text, allowed)
	if err != nil {
		return "", nil, err
	}
	if len(replies) == 0 {
		text, replies = extractBareToolCall(text, allowed)
	}
	if len(replies) == 0 {
		return text, filterToolCalls(res.ToolCalls, allowed, mode, name, parallel), nil
	}
	if !parallel {
		replies = replies[:1]
	}
	return text, replyToolCalls(replies), nil
}
