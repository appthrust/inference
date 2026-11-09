# Inference

Inference is AppThrust's GPU-backed, OpenAI-compatible chat completions service.
Its Go server supervises vLLM with the baked Qwen2.5-0.5B-Instruct model, exposes
`POST /v1/chat/completions`, `GET /v1/models`, and `GET /metrics`, and streams
completions without buffering. It is released independently of the platform.

The server listens on port 8000 and starts vLLM on `127.0.0.1:8001`. Requests are
limited to 8 MiB, four simultaneous completions, and 120 seconds. The model's
4096-token context reserves 1024 tokens for output; `/v1/models` advertises a
3072-token prompt budget. Management routes are not exposed, and edge credentials
are not forwarded upstream. `/healthz` is liveness; `/health` requires a healthy
upstream advertising the baked model; `/releasez` reports release identity.

## Develop

Use Go 1.26 or newer and Bun 1.4.1. The public Go SDK is
[`github.com/appthrust/appserver-go`](https://github.com/appthrust/appserver-go)
at `v0.3.0`, without a platform checkout or module replacement.

```sh
go build ./...
go vet ./...
go test ./...
bun test hack/
# Compile the image's Go stage without downloading the large vLLM runtime.
docker build --platform linux/amd64 --target server-builder .
```

Go and publisher tests need no GPU or live services. To use an already-running
compatible upstream locally:

```sh
INFERENCE_VLLM_BIN= INFERENCE_UPSTREAM=http://127.0.0.1:8001 go run ./cmd/server --listen :8000
```

Otherwise `INFERENCE_VLLM_BIN` defaults to `vllm`, using the baked model at
`/models/qwen2.5-0.5b-instruct`. A custom `INFERENCE_UPSTREAM` requires an explicitly
empty `INFERENCE_VLLM_BIN`; the URL must be an HTTP(S) origin without credentials,
a path, query, or fragment. The production image requires an NVIDIA GPU, bakes
weights from an immutable Hugging Face revision, runs offline as `65532:65532`,
and starts `/inference-server`. Its catalog requests one `nvidia-gpu` and a
writable scratch volume at `/home/nonroot`.

## Release

Next release: **`v0.3.0`**, upgrading the shared server SDK to v0.3.0.
The catalog publisher appends the built image; existing version digests stay immutable.

Create and push a new SemVer tag, starting with `v0.2.0` because the compatibility
floor already owns `0.1.0`:

```sh
git tag v0.2.0
git push origin v0.2.0
```

`.github/workflows/publish.yml` builds Linux amd64 and arm64 on native Blacksmith
runners, not QEMU: the runtime stage executes target-architecture Python to bake
the model. Each job pushes `ghcr.io/appthrust/inference:<version>-<arch>` and saves
its digest. The publication job creates `ghcr.io/appthrust/inference:<version>`
from those immutable digests, resolves the index with ORAS, and signs it with
Sigstore keyless signing. Make the GHCR package public after its first push.
`GITHUB_TOKEN` needs `packages:write`; signing needs `id-token:write`.

Configure repository secret `CLOUD_CATALOG_ADMIN_TOKEN` and variable
`CLOUD_API_URL` (default `https://api.appthrust.com`). Grant access to organization
secrets `APPTHRUST_GITHUB_ACTIONS_APP_ID` and
`APPTHRUST_GITHUB_ACTIONS_APP_PRIVATE_KEY`. The GitHub App must be installed on
`appthrust/cloud` with Contents write permission for repository dispatches.

The publisher's `submit` command merges persisted version history with
`catalog/entry.yaml`, PUTs `/api/v1/catalog/entries/inference`, and POSTs
`/api/v1/catalog/publish`. A `queued` response needs no dispatch; a `snapshot`
response triggers the Cloud `catalog-publish` repository event using a short-lived
GitHub App token. Cloud requires its own `CLOUD_CATALOG_API_URL` variable and
`CLOUD_CATALOG_ADMIN_TOKEN` secret.

The `await` command needs no admin token. It polls `ghcr.io/appthrust/catalog:latest`
every 15 seconds for at most 10 minutes and requires the exact Inference version
and image digest in its verified index layer. It verifies the catalog signature
against GitHub's OIDC issuer and the Cloud `catalog-publish.yml` workflow on
`main`; dispatch alone is not success. Manual publication uses the same commands
and requires ORAS 1.3.0 and Cosign 3.1.3:

```sh
bun hack/publish-entry.ts submit --version 0.2.0 --digest sha256:<image-index-digest>
# Only if submit returns state=snapshot, use an installation token for Cloud:
gh api -X POST repos/appthrust/cloud/dispatches -f event_type=catalog-publish -f 'client_payload[snapshot_id]=<snapshot-id>'
bun hack/publish-entry.ts await --version 0.2.0 --digest sha256:<image-index-digest>
```

## Catalog and image contract

- `catalog/entry.yaml` is the complete ApplicationStackDefinition, including the
  unchanged `0.1.0` seed image as a compatibility floor. New versions use
  `ghcr.io/appthrust/inference@sha256:<index-digest>` and stream `<major.minor>`.
- Image digests are immutable per version. Repeating a version/digest is
  idempotent; replacing its digest fails. Earlier versions remain served, and
  only the newest published version is default. Workflow concurrency serializes
  the read/merge/PUT sequence across tags.
- Moving an installation from the seed image repository requires an explicit
  version update with `replaceRepository: true`. Afterwards `follow` tracks the
  selected stream; `pinned` requires explicit version updates.
- `/releasez` uses catalog-rendered `INFERENCE_RELEASE_VERSION` and
  `INFERENCE_ARTIFACT_DIGEST`, optionally `INFERENCE_SOURCE_REVISION`, or the
  JSON file selected by `INFERENCE_RELEASE_FILE`.

Original server sources: platform `cmd/inference-server/`. The catalog definition
comes from Cloud's Inference seed; the publisher follows the Directory app template.
