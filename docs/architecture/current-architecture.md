# Current MLX-One architecture

This diagram describes the local `release/v0.1.4` candidate, including the
GPT-OSS serving and UI changes prepared for maintainer publication. Arrows show calls or execution
dependencies. Dashed arrows show optional execution or shared services.

```mermaid
flowchart TB
    subgraph clients[Applications and clients]
        UI["Bundled Svelte chat UI"]
        AGENTS["Coding agents / OpenAI-compatible clients"]
        APPS["Python applications"]
        TERMINAL["Terminal"]
    end

    subgraph interfaces[MLX-One interfaces]
        HTTP["FastAPI server<br/>HTTP APIs + SSE streaming"]
        SDK["Python API<br/>Engine + modality APIs"]
        CLI["mlx-one CLI"]
    end

    subgraph runtime[Runtime and workflows]
        SERVE["Serving coordination<br/>Authentication, model lifecycle, queue"]
        EXEC["Native generation and task services<br/>Reference execution / eligible actor runners"]
        LOAD["Lazy registry + checkpoint loaders<br/>Local files / Hugging Face artifacts"]
        STATE["Cache and memory management<br/>KV state, prefix reuse, admission guards"]
        BOOST["MLXBoost — optional<br/>GPT-OSS recipes + gated experiments"]
        TRAIN["Native LoRA / QLoRA SFT<br/>Adapter export"]
        EVIDENCE["Evaluation and qualification<br/>Benchmarks, comparisons, evidence records"]
    end

    subgraph native[Native model families — seven modality groups]
        TEXT["Text generation<br/>GPT-OSS, Qwen, LFM, GPT-2, OpenELM, T5"]
        VLM["Vision-language"]
        AUDIO["Audio and speech"]
        OCR["OCR and documents"]
        RETRIEVAL["Embeddings and reranking"]
        SEG["Segmentation and tracking"]
        CLASSIFY["Classification and NER"]
        OPS["Shared native operations<br/>Attention, RoPE, MoE, normalization, quantization"]
    end

    subgraph compute[Apple execution stack]
        MLX["Apple MLX<br/>Tensor execution + autodiff"]
        METAL["Metal GPU backend"]
        APPLE["Apple Silicon<br/>Unified memory"]
    end

    UI --> HTTP
    AGENTS --> HTTP
    APPS --> SDK
    TERMINAL --> CLI
    CLI -->|serve| HTTP
    HTTP --> SERVE --> EXEC
    SDK --> EXEC
    CLI --> EXEC
    SDK --> TRAIN
    CLI --> TRAIN
    CLI --> EVIDENCE
    EXEC --> LOAD
    TRAIN --> LOAD
    EXEC -.-> STATE
    EXEC -.->|explicit opt-in| BOOST
    BOOST -.->|eligible GPT-OSS path| TEXT
    EXEC --> TEXT & VLM & AUDIO & OCR & RETRIEVAL & SEG & CLASSIFY
    TRAIN --> TEXT
    EVIDENCE -.-> EXEC
    TEXT & VLM & AUDIO & OCR & RETRIEVAL & SEG & CLASSIFY --> OPS
    OPS --> MLX
    TRAIN --> MLX
    MLX --> METAL --> APPLE

    classDef optional fill:#fff3d6,stroke:#c38b22,color:#392900;
    classDef foundation fill:#e5f5ef,stroke:#009b87,color:#123c33;
    class BOOST optional;
    class MLX,METAL,APPLE foundation;
```

Each modality has task-specific preprocessing, loading and output handling;
they do not all execute through the public text `Engine`. The runtime node
groups the text engine, server generation engine and native task services.

Cache layouts, batching, prefix reuse and acceleration require matching model
capabilities and settings. MLXBoost ships inside MLX-One and requires explicit
opt-in. Experimental kernels are not automatically enabled or qualified across
all models. This diagram is an implementation map, not a release-support matrix
or a performance guarantee.

Primary source boundaries: `src/mlx_one/__init__.py`, `cli.py`, `server/`,
`engine/`, `models/`, `mlxboost/`, `training.py` and `ui/`.
