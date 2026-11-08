"""Serving metadata and leased operations for native fixed-vector retrieval."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalServingBundle:
    retrieval: object
    chat_template: object = None
    parameter_count: object = None
    quantization: object = None

    def __getattr__(self, name):
        return getattr(self.retrieval, name)

    @property
    def context_length(self):
        config = self.retrieval.model.config
        return getattr(config, "sentence_max_length", getattr(config, "max_seq_length", 8192))


def serving_operation(manager, task):
    """Acquire the model lease inside the scheduled operation until it finishes."""

    def run(*args, **kwargs):
        from mlx_one.retrieval import embed, rerank

        with manager.model_lease():
            bundle = manager.current_model().retrieval
            return (embed if task == "embedding" else rerank)(bundle, *args, **kwargs)

    return run
