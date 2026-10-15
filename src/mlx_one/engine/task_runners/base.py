"""Task adapter description/cost/packing independent of model math and scheduling."""
from __future__ import annotations

from dataclasses import dataclass

from mlx_one.engine.contracts import TaskKind, WorkItem
from mlx_one.engine.tasks import PreparedTask


@dataclass(frozen=True)
class TaskAdapterSpec:
    architecture: str
    tasks: tuple[TaskKind, ...]
    version: str = 'native-task-adapter-v1'
    persistent_decode_state: bool = False
    checkpoint_qualified: bool = False


class NativeTaskAdapter:
    family=None
    supported_tasks=()

    def describe(self,bundle):
        if bundle is not self.bundle:
            raise ValueError('task adapter is bound to another runtime')
        return TaskAdapterSpec(self.family,self.supported_tasks)

    def estimate(self,prepared,plan=None):
        if not isinstance(prepared,PreparedTask):
            raise ValueError('task estimate requires prepared native input')
        return {'real_input_cost':sum(prepared.lengths),'output_bound':len(prepared.lengths),
                'transient_bytes':prepared.transient_bytes,'persistent_cache_bytes':0}

    def pack(self,work):
        if not work or any(not isinstance(w,WorkItem) or not isinstance(w.payload,PreparedTask)
                           for w in work):
            raise ValueError('task packing requires typed work and prepared inputs')
        first=work[0].compatibility
        if (len({w.request_id for w in work})!=len(work)
                or any(w.compatibility!=first or w.payload.task!=first.task for w in work)
                or first.model_instance!=str(id(self.bundle))
                or first.adapter_version!='native-task-adapter-v1'):
            raise ValueError('task batch execution identity mismatch')
        return tuple(w.payload for w in work)

    def execute(self,packed,state=None):
        return self.run_batch(packed)

    def unpack(self,output):
        return tuple(output)

    def encode(self,packed):
        return self.execute(packed)
