package proxy

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"unicode/utf8"

	"quickwork-openai/internal/config"
)

func TestQuickWorkFrameBudget(t *testing.T) {
	cfg := config.Default()
	u := NewUpstream(&cfg)
	source := strings.Repeat("中文<&\n", 10000)
	msgs := []Message{
		{Role: "user", Content: "original request"},
		{Role: "tool", ToolCallID: "one", Content: source},
		{Role: "tool", ToolCallID: "two", Content: "latest result"},
	}
	transcript, err := formatPrompt(msgs)
	if err != nil {
		t.Fatal(err)
	}
	in := ChatInput{Messages: msgs, PromptPrefix: "catalog\n", Prompt: "catalog\n" + transcript, SystemPrompt: "caller rules"}
	frame, err := u.userMessage(&Session{}, in, uuid())
	if err != nil {
		t.Fatal(err)
	}
	b, _ := json.Marshal(frame)
	p := frame["payload"].(map[string]any)
	text := p["text"].(string)
	if len(b) > maxUserFrameBytes || !utf8.ValidString(text) || p["system_prompt"] != "caller rules" || msgs[1].Content != source {
		t.Fatal("invalid frame or original history changed")
	}
	for _, want := range []string{"catalog\nUser: original request", "call_id=one", "Middle omitted", "call_id=two", "latest result"} {
		if !strings.Contains(text, want) {
			t.Fatalf("missing %q", want)
		}
	}
	in.Prompt = source
	in.Messages = []Message{{Role: "user", Content: source}}
	if _, err = u.userMessage(&Session{}, in, uuid()); !errors.Is(err, errFrameTooLarge) {
		t.Fatal("expected oversized input error", err)
	}
}

func TestFrameBudgetPreservesPendingToolResults(t *testing.T) {
	const prefix = "client tool catalog\n"
	args := strings.Repeat("old write content ", 1000)
	latest := strings.Repeat("unread result ", 200)
	msgs := []Message{
		{Role: "user", Content: "read after writing"},
		{Role: "assistant", ToolCalls: []ToolCall{{ID: "w", Function: FunctionCall{Name: "write", Arguments: args}}}},
		{Role: "tool", ToolCallID: "w", Content: "file created"},
		{Role: "assistant", ToolCalls: []ToolCall{{ID: "r", Function: FunctionCall{Name: "read", Arguments: "{}"}}}},
		{Role: "tool", ToolCallID: "r", Content: latest},
	}
	transcript, err := formatPrompt(msgs)
	if err != nil {
		t.Fatal(err)
	}
	in := ChatInput{Messages: msgs, PromptPrefix: prefix, Prompt: prefix + transcript}
	got, err := fitPrompt(in, 6000)
	if err != nil || !strings.Contains(got, latest) || !strings.Contains(got, omitted) || msgs[1].ToolCalls[0].Function.Arguments != args {
		t.Fatalf("pending result or original call changed (error %v)", err)
	}
}
