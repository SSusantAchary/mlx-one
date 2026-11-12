"""Local dense retrieval followed by paired reranking using only MLX-One."""

import argparse

import numpy as np

from mlx_one.retrieval import embed, load_retrieval_model, rerank

PINS = {
    "BAAI/bge-small-en-v1.5": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    "intfloat/multilingual-e5-small": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
    "cross-encoder/ms-marco-MiniLM-L6-v2": "233902d25c440f23af6f7d6e94d2946bac0bee0a",
    "BAAI/bge-reranker-v2-m3": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default="What is the capital of France?")
    parser.add_argument("--embedder", choices=list(PINS)[:2], default=list(PINS)[0])
    parser.add_argument("--reranker", choices=list(PINS)[2:], default=list(PINS)[2])
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    documents = [
        "The capital of France is Paris.",
        "Bananas are yellow fruit.",
        "Berlin is the capital of Germany.",
        "A compiler translates programs.",
    ]
    encoder = load_retrieval_model(
        args.embedder, revision=PINS[args.embedder], offline=args.offline
    )
    query = np.asarray(embed(encoder, args.query, input_type="query").embeddings[0])
    vectors = np.asarray(embed(encoder, documents).embeddings)
    similarities = vectors @ query / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query))
    candidates = [documents[i] for i in np.argsort(-similarities)[:3]]
    scorer = load_retrieval_model(args.reranker, revision=PINS[args.reranker], offline=args.offline)
    for item in rerank(scorer, args.query, candidates).items:
        print(f"{item.score: .6f}\t{item.document}")


if __name__ == "__main__":
    main()
