package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"syscall"
	"time"
)

const (
	servedModel = "qwen2.5-0.5b-instruct"
	modelPath   = "/models/qwen2.5-0.5b-instruct"
	upstreamURL = "http://127.0.0.1:8001"
)

type config struct {
	upstream  *url.URL
	vllmBin   string
	projectID string
}

func configFromEnv() (config, error) {
	cfg := config{vllmBin: "vllm", projectID: os.Getenv("PLATFORM_PROJECT_ID")}
	if bin, set := os.LookupEnv("INFERENCE_VLLM_BIN"); set {
		cfg.vllmBin = bin
	}
	upstream := os.Getenv("INFERENCE_UPSTREAM")
	if upstream == "" {
		upstream = upstreamURL
	}
	u, err := url.Parse(upstream)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Hostname() == "" || u.User != nil || (u.Path != "" && u.Path != "/") || u.RawQuery != "" || u.ForceQuery || u.Fragment != "" {
		return config{}, errors.New("INFERENCE_UPSTREAM must be an HTTP(S) origin without credentials, a path, query, or fragment")
	}
	u.Path, u.RawPath = "", ""
	if cfg.vllmBin != "" && u.String() != upstreamURL {
		return config{}, errors.New("INFERENCE_UPSTREAM requires an empty INFERENCE_VLLM_BIN unless it is http://127.0.0.1:8001")
	}
	cfg.upstream = u
	return cfg, nil
}

func main() {
	listen := flag.String("listen", ":8000", "HTTP listen address")
	flag.Parse()
	if flag.NArg() != 0 {
		log.Fatal("usage: inference-server [--listen :8000]")
	}
	cfg, err := configFromEnv()
	if err != nil {
		log.Fatal(err)
	}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	if err := serve(ctx, *listen, cfg); err != nil {
		log.Fatal(err)
	}
}

func serve(ctx context.Context, listen string, cfg config) error {
	listener, err := net.Listen("tcp", listen)
	if err != nil {
		return fmt.Errorf("listen: %w", err)
	}
	s := newSupervisor(cfg.vllmBin, cfg.upstream.String())
	childCtx, cancelChild := context.WithCancel(context.Background())
	defer cancelChild()
	childDone := make(chan struct{})
	go func() {
		defer close(childDone)
		s.Run(childCtx)
	}()
	server := &http.Server{
		Handler:           newHandler(cfg.upstream, s.Ready),
		ReadHeaderTimeout: 10 * time.Second, ReadTimeout: requestTimeout,
		WriteTimeout: requestTimeout + 5*time.Second, IdleTimeout: 60 * time.Second,
	}
	served := make(chan error, 1)
	go func() { served <- server.Serve(listener) }()
	log.Printf("inference-server listening on %s (project %s)", listener.Addr(), cfg.projectID)
	select {
	case <-ctx.Done():
	case err = <-served:
	}
	s.alive.Store(false)
	shutdown, cancelShutdown := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancelShutdown()
	if shutdownErr := server.Shutdown(shutdown); shutdownErr != nil {
		log.Printf("HTTP shutdown: %v", shutdownErr)
		_ = server.Close()
	}
	cancelChild()
	<-childDone
	if errors.Is(err, http.ErrServerClosed) {
		return nil
	}
	return err
}
