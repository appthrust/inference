package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"sync/atomic"
	"syscall"
	"time"
)

const (
	maxModelLen = 4096
	// Advertise input capacity separately from the context window so Ask leaves
	// at least 1024 tokens for generated output. Keep this tied to vLLM's flag.
	reservedOutputTokens = 1024
	maxPromptTokens      = maxModelLen - reservedOutputTokens
)

type supervisor struct {
	bin          string
	upstream     string
	client       *http.Client
	alive        atomic.Bool
	warmed       atomic.Bool
	pollInterval time.Duration
	restartDelay time.Duration
	stopTimeout  time.Duration
}

func newSupervisor(bin, upstream string) *supervisor {
	return &supervisor{
		bin: bin, upstream: upstream, client: upstreamClient(),
		pollInterval: time.Second, restartDelay: time.Second, stopTimeout: 10 * time.Second,
	}
}

func (s *supervisor) Ready() bool { return s.alive.Load() && s.warmed.Load() }

func vllmArgs() []string {
	return []string{
		"serve", modelPath,
		"--served-model-name", servedModel,
		"--gpu-memory-utilization", "0.5",
		"--max-model-len", strconv.Itoa(maxModelLen),
		"--host", "127.0.0.1", "--port", "8001",
	}
}

// Run owns and reaps the child process group, including model engine workers.
// An explicitly empty binary uses an external upstream for focused tests.
func (s *supervisor) Run(ctx context.Context) {
	defer s.alive.Store(false)
	defer s.warmed.Store(false)
	if s.bin == "" {
		s.alive.Store(true)
		s.monitor(ctx)
		return
	}
	backoff := s.restartDelay
	for ctx.Err() == nil {
		cmd := exec.Command(s.bin, vllmArgs()...)
		cmd.Env = append(os.Environ(), "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1")
		cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
		cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
		started := time.Now()
		if err := cmd.Start(); err != nil {
			log.Printf("starting vllm: %v", err)
		} else {
			s.alive.Store(true)
			childCtx, cancel := context.WithCancel(ctx)
			monitored := make(chan struct{})
			go func() {
				defer close(monitored)
				s.monitor(childCtx)
			}()
			exited := make(chan error, 1)
			go func() { exited <- cmd.Wait() }()
			select {
			case err := <-exited:
				s.alive.Store(false)
				_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
				log.Printf("vllm exited: %v", err)
			case <-ctx.Done():
				s.alive.Store(false)
				cancel()
				s.stop(cmd, exited)
			}
			cancel()
			<-monitored
			s.warmed.Store(false)
			if time.Since(started) >= time.Minute {
				backoff = s.restartDelay
			}
		}
		if !waitContext(ctx, backoff) {
			return
		}
		backoff = min(backoff*2, 30*time.Second)
	}
}

func (s *supervisor) stop(cmd *exec.Cmd, exited <-chan error) {
	_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGTERM)
	timer := time.NewTimer(s.stopTimeout)
	defer timer.Stop()
	select {
	case <-exited:
	case <-timer.C:
		_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
		<-exited
	}
	_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
}

func (s *supervisor) monitor(ctx context.Context) {
	defer s.warmed.Store(false)
	for ctx.Err() == nil {
		if err := s.probe(ctx); err != nil {
			s.warmed.Store(false)
		} else if !s.warmed.Load() {
			if err := s.warm(ctx); err != nil {
				if ctx.Err() == nil {
					log.Printf("waiting for inference model: %v", err)
				}
			} else if ctx.Err() == nil {
				s.warmed.Store(true)
				log.Printf("inference model %s ready", servedModel)
			}
		}
		if !waitContext(ctx, s.pollInterval) {
			return
		}
	}
}

func (s *supervisor) probe(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, s.upstream+"/health", nil)
	if err != nil {
		return err
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if _, err := io.Copy(io.Discard, io.LimitReader(resp.Body, 1<<20)); err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("vllm health returned HTTP %d", resp.StatusCode)
	}
	return nil
}

// vLLM loads and warms the engine before serving the model catalog. A successful
// catalog response naming the baked model, not just a live HTTP process, is ready.
func (s *supervisor) warm(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, s.upstream+"/v1/models", nil)
	if err != nil {
		return err
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("vllm model readiness returned HTTP %d", resp.StatusCode)
	}
	var result struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&result); err != nil {
		return err
	}
	for _, model := range result.Data {
		if model.ID == servedModel {
			return nil
		}
	}
	return errors.New("vllm model readiness must include qwen2.5-0.5b-instruct")
}

func waitContext(ctx context.Context, delay time.Duration) bool {
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-timer.C:
		return ctx.Err() == nil
	}
}
