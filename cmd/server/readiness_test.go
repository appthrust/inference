package main

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestReadinessRequiresModelsAndRecoversAfterOutage(t *testing.T) {
	var healthy, modelReady atomic.Bool
	var modelProbes atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			if !healthy.Load() {
				w.WriteHeader(http.StatusServiceUnavailable)
			}
		case "/v1/models":
			modelProbes.Add(1)
			if !modelReady.Load() {
				w.WriteHeader(http.StatusServiceUnavailable)
				return
			}
			_, _ = io.WriteString(w, `{"data":[{"id":"qwen2.5-0.5b-instruct"}]}`)
		default:
			t.Errorf("unexpected readiness request: %s", r.URL.Path)
		}
	}))
	defer upstream.Close()
	s := newSupervisor("", upstream.URL)
	s.pollInterval = 5 * time.Millisecond
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { defer close(done); s.Run(ctx) }()
	defer func() { cancel(); <-done }()
	healthy.Store(true)
	require.Eventually(t, func() bool { return modelProbes.Load() > 0 }, 5*time.Second, time.Millisecond)
	require.False(t, s.Ready(), "a healthy process without the model is not ready")
	modelReady.Store(true)
	require.Eventually(t, s.Ready, 5*time.Second, time.Millisecond)
	h := handlerForTest(t, upstream.URL, s.Ready)
	require.Equal(t, http.StatusOK, requestForTest(h, http.MethodGet, "/health", "").Code)
	healthy.Store(false)
	require.Eventually(t, func() bool { return !s.Ready() }, 5*time.Second, time.Millisecond)
	modelReady.Store(false)
	probesBeforeRecovery := modelProbes.Load()
	healthy.Store(true)
	require.Eventually(t, func() bool { return modelProbes.Load() > probesBeforeRecovery }, 5*time.Second, time.Millisecond)
	require.False(t, s.Ready(), "outage recovery must recheck the model")
	modelReady.Store(true)
	require.Eventually(t, s.Ready, 5*time.Second, time.Millisecond)
	cancel()
	<-done
	require.False(t, s.Ready())
}

func TestModelReadinessRejectsMissingModelAndMalformedResponses(t *testing.T) {
	for _, body := range []string{`{}`, `{"data":[]}`, `{"data":[{"id":"another-model"}]}`, `not json`} {
		upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_, _ = io.WriteString(w, body)
		}))
		s := newSupervisor("", upstream.URL)
		require.Error(t, s.warm(context.Background()))
		require.False(t, s.Ready())
		upstream.Close()
	}
}

func TestVLLMArgumentsAreReleaseOwned(t *testing.T) {
	require.Equal(t, []string{
		"serve", "/models/qwen2.5-0.5b-instruct",
		"--served-model-name", "qwen2.5-0.5b-instruct",
		"--gpu-memory-utilization", "0.5",
		"--max-model-len", "4096",
		"--host", "127.0.0.1", "--port", "8001",
	}, vllmArgs())
}
