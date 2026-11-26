# ADR 0005: vLLM is called over HTTP

Status: Accepted

## Context
The worker needs a real inference engine. vLLM could be imported as a library or
run as a server. `CLAUDE.md` says InferScale must not duplicate vLLM's scheduling,
batching or KV cache.

## Decision
`VllmBackend` calls a vLLM OpenAI-compatible server (`POST /v1/completions`,
`GET /health`) with httpx. The worker does not import `vllm` and adds no
concurrency limit of its own.

## Consequences
- No new Python dependency, and no CUDA or torch in the worker image.
- The worker package does not need to be split (answers the open point in ADR 0001).
- Error mapping keeps the Router's rule: a request that never reached vLLM
  (connect error, 503) raises `BackendUnavailableError` and may fail over. A read
  timeout or dropped connection raises `BackendError` and is not retried.
- Only an allow-list of sampling parameters is forwarded. Other keys are dropped
  with a warning.
- The request `model` is forwarded unchanged. It must equal the model name the
  vLLM server serves, and the Router's worker config must list that name.
- Streaming and chat templates are not supported.
- Assumptions not verified without a real server: the `/health` and
  `/v1/completions` shapes, `top_k` support, and that vLLM aborts a generation
  when the client disconnects. `scripts/vllm_check.py` checks the first two.
