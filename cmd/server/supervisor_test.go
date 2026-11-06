package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

type helperEvent struct {
	Kind string `json:"kind"`
	PID  int    `json:"pid"`
}

// This test is also the fake vllm executable. The shell wrapper execs the
// current test binary, so tests exercise real process exit and signal handling
// without compiling a second program or depending on an installed VLLM.
func TestVLLMHelper(t *testing.T) {
	if os.Getenv("INFERENCE_TEST_VLLM") != "1" {
		return
	}
	if strings.Join(os.Args[len(os.Args)-len(vllmArgs()):], " ") != strings.Join(vllmArgs(), " ") || os.Getenv("HF_HUB_OFFLINE") != "1" || os.Getenv("TRANSFORMERS_OFFLINE") != "1" {
		fmt.Fprintln(os.Stderr, "helper requires pinned model arguments and offline mode")
		os.Exit(2)
	}
	event := func(kind string) {
		payload := fmt.Sprintf(`{"kind":%q,"pid":%d}`, kind, os.Getpid())
		client := &http.Client{Timeout: 3 * time.Second}
		resp, err := client.Post(os.Getenv("INFERENCE_TEST_EVENTS"), "application/json", strings.NewReader(payload))
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		_ = resp.Body.Close()
	}
	terminated := make(chan os.Signal, 1)
	signal.Notify(terminated, syscall.SIGTERM)
	listener, err := net.Listen("tcp", os.Getenv("INFERENCE_TEST_LISTEN"))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{}`)
	})
	mux.HandleFunc("GET /v1/models", func(w http.ResponseWriter, _ *http.Request) {
		event("warm")
		_, _ = io.WriteString(w, `{"data":[{"id":"qwen2.5-0.5b-instruct"}]}`)
	})
	server := &http.Server{Handler: mux, ReadHeaderTimeout: time.Second}
	event("start")
	go func() { _ = server.Serve(listener) }()
	<-terminated
	event("term")
	_ = server.Close()
	os.Exit(0)
}

func TestSupervisorRestartsAndForwardsSIGTERM(t *testing.T) {
	events := make(chan helperEvent, 16)
	callback := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var event helperEvent
		if err := json.NewDecoder(r.Body).Decode(&event); err != nil {
			t.Errorf("decoding helper event: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		events <- event
		w.WriteHeader(http.StatusNoContent)
	}))
	defer callback.Close()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	address := listener.Addr().String()
	require.NoError(t, listener.Close())
	executable, err := os.Executable()
	require.NoError(t, err)
	t.Setenv("INFERENCE_TEST_VLLM", "1")
	t.Setenv("INFERENCE_TEST_EXECUTABLE", executable)
	t.Setenv("INFERENCE_TEST_LISTEN", address)
	t.Setenv("INFERENCE_TEST_EVENTS", callback.URL)
	// The supervisor must force offline operation even if the parent disables it.
	t.Setenv("HF_HUB_OFFLINE", "0")
	t.Setenv("TRANSFORMERS_OFFLINE", "0")
	script := filepath.Join(t.TempDir(), "vllm")
	require.NoError(t, os.WriteFile(script, []byte("#!/bin/sh\nexec \"$INFERENCE_TEST_EXECUTABLE\" -test.run '^TestVLLMHelper$' -- \"$@\"\n"), 0700))
	s := newSupervisor(script, "http://"+address)
	s.pollInterval = 5 * time.Millisecond
	s.restartDelay = 200 * time.Millisecond
	s.stopTimeout = 2 * time.Second
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { defer close(done); s.Run(ctx) }()
	defer func() {
		cancel()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Error("supervisor failed to stop")
		}
	}()
	nextEvent := func(kind string) helperEvent {
		t.Helper()
		select {
		case event := <-events:
			require.Equal(t, kind, event.Kind)
			return event
		case <-time.After(5 * time.Second):
			t.Fatalf("no %s event from child", kind)
			return helperEvent{}
		}
	}
	first := nextEvent("start")
	require.Equal(t, first.PID, nextEvent("warm").PID)
	require.Eventually(t, s.Ready, 5*time.Second, time.Millisecond)
	crashed := time.Now()
	require.NoError(t, syscall.Kill(first.PID, syscall.SIGKILL))
	require.Eventually(t, func() bool { return !s.Ready() }, time.Second, time.Millisecond)
	second := nextEvent("start")
	require.NotEqual(t, first.PID, second.PID)
	require.GreaterOrEqual(t, time.Since(crashed), s.restartDelay)
	require.Equal(t, second.PID, nextEvent("warm").PID)
	require.Eventually(t, s.Ready, 5*time.Second, time.Millisecond)
	cancel()
	require.Equal(t, second.PID, nextEvent("term").PID)
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("supervisor did not reap terminated child")
	}
	require.False(t, s.Ready())
	require.ErrorIs(t, syscall.Kill(second.PID, 0), syscall.ESRCH)
	require.Empty(t, events, "shutdown must not restart VLLM")
}
