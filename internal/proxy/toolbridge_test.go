package proxy

import (
	"encoding/json"
	"errors"
	"testing"
)

func TestResolveToolCalls(t *testing.T) {
	const marker = `<client_tool_call>{"name":"lookup","arguments":{"sku":"A"}}</client_tool_call>`
	allowed := map[string]bool{"lookup": true}
	native := []ToolCallState{{ID: "internal", Name: "search"}, {ID: "client", Name: "lookup", Arguments: `{"sku":"B"}`}}
	for _, tc := range []struct{ name, text, wantText, args string }{
		{"bridge wins", marker, "", `{"sku":"A"}`},
		{"native with prose", "checking", "checking", `{"sku":"B"}`},
		{"native only", "", "", `{"sku":"B"}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			text, calls, err := resolveToolCalls(ChatResult{Text: tc.text, ToolCalls: native}, allowed, "auto", "", true)
			if err != nil || text != tc.wantText || len(calls) != 1 || calls[0].Function.Name != "lookup" || calls[0].Function.Arguments != tc.args {
				t.Fatalf("text %q, calls %+v", text, calls)
			}
		})
	}
	_, calls, err := resolveToolCalls(ChatResult{Text: marker + marker}, allowed, "auto", "", false)
	if err != nil || len(calls) != 1 {
		t.Fatalf("single call requested, got %d", len(calls))
	}
	text, calls, err := resolveToolCalls(ChatResult{Text: marker, ToolCalls: native}, allowed, "none", "", true)
	if err != nil || text != marker || len(calls) != 0 {
		t.Fatalf("disabled tools: %q %+v", text, calls)
	}
}

func TestMarkedBashCommand(t *testing.T) {
	const want = `for f in docs/*.md; do echo "==== $f"; sed -n '1,45p' "$f"; done
grep -nE '^\s*--[a-z-]+:' styles.css | head -60`
	const raw = `{"name":"bash","arguments":{"workdir":"/tmp","command":"for f in docs/*.md; do echo \"==== $f\"; sed -n '1,45p' \"$f\"; done
grep -nE '^\s*--[a-z-]+:' styles.css | head -60"}}`
	for _, tc := range []struct{ name, payload, command string }{
		{"regex and multiline", raw, want},
		{"valid JSON escapes", `{"name":"bash","arguments":{"command":"printf '%s\\n' \"$f\""}}`, `printf '%s\n' "$f"`},
		{"literal closing marker", `{"name":"bash","arguments":{"command":"echo '</client_tool_call>'"}}`, `echo '</client_tool_call>'`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			result := ChatResult{Text: "读取样式。\n" + clientToolCallOpen + tc.payload + clientToolCallClose}
			text, calls, err := resolveToolCalls(result, map[string]bool{"bash": true}, "auto", "", true)
			if err != nil || text != "读取样式。" || len(calls) != 1 || calls[0].Function.Name != "bash" {
				t.Fatalf("%q %+v %v", text, calls, err)
			}
			var args struct{ Command string }
			if err := json.Unmarshal([]byte(calls[0].Function.Arguments), &args); err != nil || args.Command != tc.command {
				t.Fatalf("command %q, want %q (%v)", args.Command, tc.command, err)
			}
		})
	}
}

func TestInvalidToolMarkerFailsWholeTurn(t *testing.T) {
	const ok = `<client_tool_call>{"name":"bash","arguments":{}}</client_tool_call>`
	for _, bad := range []string{
		`<client_tool_call>{"name":"bash","arguments":{"command":"echo "$f""}}</client_tool_call>`,
		`<client_tool_call>{"name":"bash","arguments":{`,
		`<client_tool_call>{"name":"bash","arguments":"broken"}</client_tool_call>`,
		`<client_tool_call>{"name":"undeclared","arguments":{}}</client_tool_call>`,
	} {
		text, calls, err := resolveToolCalls(ChatResult{Text: ok + bad}, map[string]bool{"bash": true}, "auto", "", true)
		if !errors.Is(err, errToolMarker) || text != "" || len(calls) != 0 {
			t.Fatalf("unexpected partial response: %q %+v %v", text, calls, err)
		}
	}
}
