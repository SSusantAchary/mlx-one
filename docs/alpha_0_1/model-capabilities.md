# Registry-backed model/task capabilities

VERIFIED_NO_CHANGE to registry source: existing ModelRegistration, RuntimeRegistration/TaskSpec, `runtime_coverage()`, `registry_inventory()` and exact-scope CapabilityRegistry/InferenceEvidenceRegistry already provide lazy architecture, integration and evidence dimensions. No second registry or qualification runner added. `benchmarks/native_tasks.py inventory` is the existing backend-free descriptor command; `registry` CLI inspects evidence-backed operation support. Architecture registration never grants checkpoint or alpha support.

| Native task | Existing Python API | Existing HTTP task route | Preprocessing / limits |
|---|---|---|---|
| text-generation | Engine.generate/stream; generate/stream_generate | /v1/chat/completions | tokenizer/chat/context, checkpoint-dependent |
| vision-language-generation | load_vlm_model/stream_vlm | multimodal /v1/chat/completions | exact processor/pixels/tokens, bounded image resolver |
| embedding | load_retrieval_model/embed | /v1/embeddings | tokenizer/mask/pooling/native dimensions |
| reranking | rerank | /v1/rerank | pair/instruction/answer-token contract |
| transcription | transcribe | /v1/audio/transcriptions | PCM/codec/rate; upload/decoder limits |
| alignment / late-interaction / segmentation/video | existing task-specific APIs | no new route promised | preserve distinct tasks, not ordinary vectors |

Actual registered tasks and adapter integration are exported into support-matrix.json from task_inventory, with qualification_promoted=false. Missing checkpoint, weight/activation/KV dtype, input limits, hardware or MLX evidence stays unobserved; defaults cannot fill a verified cell. Use describe_precision for inspected ModelSpec metadata. Every architecture retains checkpoint_load_pass=false, task_quality_pass=false, hardware_pass=false and alpha_supported=false in the alpha matrix until exact current-source receipt verification. Existing validators reject stale/wrong-source/mismatched task evidence. No historical receipt attached as new proof.

Validation: lazy registry and plugin boundaries, task mismatch/aliases, schema compatibility, evidence checksum/identity, task gates and inspection: **81 passed**, 4.61s. Exact command and JUnit/log: `a3.log`, `a3.junit.xml` in ignored session; test files are runtime_registry, adapter_registry, task_contracts, registry_plugins, task_qualification and inspection. Registration count remains 27. Imports do not initialize MLX/Metal. No public API removed.
