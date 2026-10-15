"""Backend-free synthetic template; deliberately ineligible for native qualification."""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from mlx_one.engine.contracts import StageKind, TaskKind, TaskSpec
from mlx_one.engine.task_runners.base import NativeTaskAdapter
from mlx_one.engine.tasks import PreparedTask
from mlx_one.retrieval.schemas import EmbeddingResult


@dataclass(frozen=True)
class Config:
    model_type: str = 'example_encoder'
    hidden_size: int = 2

    @classmethod
    def from_dict(cls,data):
        return cls(**data)


class Model:
    def __init__(self,config):
        self.config=config
        self.batches=[]

    def parameters(self):
        import numpy as np

        return {'weight':np.ones((2,2),dtype=np.float32)}

    def __call__(self,texts):
        from mlx_one.engine.device_executor import DeviceExecutor

        assert DeviceExecutor.shared().is_owner
        self.batches.append(len(texts))
        return tuple((float(len(text)),float(sum(map(ord,text)))) for text in texts)


def load_model(path,**settings):
    return SimpleNamespace(model=Model(Config()),architecture='example_encoder',
                           path=Path(path),model_id='synthetic/example',revision='test')


def task_specs(architecture):
    return (TaskSpec(TaskKind.EMBEDDING,'encoder',(StageKind.ENCODE,),'length-buckets',
                     output_schema='embedding-v1'),)


def state_specs(architecture):
    return ()


class Adapter(NativeTaskAdapter):
    family='example_encoder'
    supported_tasks=(TaskKind.EMBEDDING,)

    def __init__(self,bundle):
        self.bundle=bundle

    def prepare(self,task,payload):
        values=payload['texts']
        values=(values,) if isinstance(values,str) else tuple(values)
        return PreparedTask(TaskKind(task),values,tuple(max(1,len(t)) for t in values))

    def run_batch(self,prepared):
        texts=tuple(text for item in prepared for text in item.payload)
        output=iter(self.bundle.model(texts))
        return tuple(EmbeddingResult(p.payload,tuple(next(output) for _ in p.payload),
            2,'document',self.bundle.model_id,self.bundle.revision) for p in prepared)
