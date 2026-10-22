# HTTP compatibility — revamp increment

Single startup model; existing bearer-key policy applies to generation/discovery/runtime. Version stays 0.1.0a1. Existing multimodal endpoints and public Python APIs remain available. No checkpoint is newly release-qualified.

| Route | Behavior |
|---|---|
| GET /health | Existing MLX health response |
| GET /v1/models | Existing single-model discovery, architecture/revision/task metadata |
| POST /v1/completions | New B1 raw text generation; no chat template required |
| POST /v1/chat/completions | Existing messages/chat-template/reasoning and SSE contract |
| GET /v1/runtime | Existing model, scheduler, cache and observed runtime metadata |
| POST /v1/embeddings, /v1/rerank, /v1/audio/transcriptions | Preserved existing service contracts |

Raw completions accept model, one nonempty string prompt, max_tokens (1..32768), temperature, top_p, optional top_k, seed, stop (one string or at most eight bounded strings), and stream. Unknown fields, prompt arrays, n, echo, suffix and logprobs requests are rejected; this is an explicit B1 subset. Both prompt and requested output must fit the loaded model's configured/native context. Unknown model returns 404, malformed request 422, context/capability error 400, capacity rejection 503, full queue 429 and scheduler timeout 408. HTTP and Python may retain their existing context-budget semantics; HTTP never silently extends positional limits.

Nonstream response uses object=text_completion and choices[{index,text,logprobs:null,finish_reason}], standard usage fields, plus mlx metrics. SSE uses data: JSON with text_completion choices and a final data: [DONE]; no assistant role, message rendering or reasoning filtering is injected into raw text. Streaming failures after headers use the existing error-event shape. Client disconnect closes scheduled work. No batching/optimization backend is newly promoted.

```sh
curl http://127.0.0.1:8181/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<loaded-model-id>","prompt":"The capital of France is","max_tokens":16,"temperature":0,"stream":true}'
```

Validation and limits: [API qualification](../qualification/api-qualification.md). Historical alpha documents retain their original route scopes and source identities; this document describes the new additive route.
