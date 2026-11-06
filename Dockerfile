# syntax=docker/dockerfile:1

FROM --platform=$BUILDPLATFORM docker.io/golang:1.27.3 AS server-builder
WORKDIR /workspace
ARG TARGETOS
ARG TARGETARCH
COPY go.mod go.sum ./
RUN --mount=type=cache,target=/go/pkg/mod go mod download
COPY cmd/server ./cmd/server
RUN --mount=type=cache,target=/root/.cache/go-build --mount=type=cache,target=/go/pkg/mod \
    CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} \
    go build -trimpath -ldflags='-s -w' -o /out/inference-server ./cmd/server

# Immutable multi-architecture index; CUDA 13.0 / vLLM 0.29.1rc1.dev422+gd05da62e9.
# This exact upstream image served Qwen on Monstera's RTX 5090 (ADR-0045 PoC).
FROM vllm/vllm-openai@sha256:f29125bcc6d276ed38a67c9dfd4e51ccbaba09ad92847a0913c0df38d9c05c71
# Apache-2.0 model and tokenizer are baked from one immutable HF revision.
# Downloading needs no GPU; runtime is forced offline and never pulls weights.
RUN python3 -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775", local_dir="/models/qwen2.5-0.5b-instruct", allow_patterns=["*.safetensors", "*.json", "*.txt", "LICENSE", "README.md"])' \
    && chmod -R a+rX /models \
    && mkdir -p /home/nonroot/.cache \
    && chown -R 65532:65532 /home/nonroot
ENV HOME=/home/nonroot \
    XDG_CACHE_HOME=/home/nonroot/.cache \
    HF_HOME=/home/nonroot/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1
COPY --from=server-builder /out/inference-server /inference-server
USER 65532:65532
EXPOSE 8000
ENTRYPOINT ["/inference-server"]
