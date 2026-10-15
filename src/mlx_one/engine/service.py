"""Task-neutral service bridge; text uses existing Engine and result/event types."""
from __future__ import annotations

import threading

from mlx_one.engine.model_pool import ModelPool


class InferenceService:
    def __init__(self, *, pool=None, batching_config=None):
        self.pool = pool or ModelPool()
        self._models = {}
        self._task_models = {}
        self._task_actor = None
        self._batching_config = batching_config
        self._lock = threading.RLock()
        self._closed = False

    def register_text(self, name, model, *, estimated_bytes=0, **settings):
        """Register local loading policy, without loading weights or executing plugins."""
        with self._lock:
            if self._closed or not name or name in self._models or name in self._task_models:
                raise ValueError("service is closed or model alias is already registered")
            self._models[name] = (model, dict(settings), estimated_bytes)

    def _lease(self, name):
        from mlx_one.engine.cache import array_nbytes
        from mlx_one.engine.public import Engine

        with self._lock:
            if self._closed:
                raise RuntimeError("inference service is closed")
            model, settings, estimate = self._models[name]
        return self.pool.acquire(name, lambda: Engine(model, **settings), estimated_bytes=estimate,
            measure=lambda engine: array_nbytes(engine.model.model.parameters()),
            drain=lambda engine: engine.close())

    def generate(self, model, prompt, *, options=None, **settings):
        with self._lease(model) as runtime:
            return runtime.value.generate(prompt, options=options, **settings)

    def stream(self, model, prompt, *, options=None, cancel=None, **settings):
        event = cancel or threading.Event()
        with self._lease(model) as runtime:
            stream = runtime.value.stream(prompt, options=options, cancel=event, **settings)
            try:
                yield from stream
            finally:
                event.set()
                stream.close()

    def register_task(self,name,model,*,loader=None,estimated_bytes=0,**settings):
        with self._lock:
            if self._closed or not name or name in self._models or name in self._task_models:
                raise ValueError("service is closed or model alias is already registered")
            self._task_models[name]=(model,loader,estimated_bytes,dict(settings))

    def _prepare_task(self,name,task,payload):
        from mlx_one.engine.actor import InferenceActor
        from mlx_one.engine.adapter_registry import get_runtime_registration
        from mlx_one.engine.cache import array_nbytes
        from mlx_one.engine.contracts import canonical_task

        task=canonical_task(task)
        with self._lock:
            if self._closed:
                raise RuntimeError("inference service is closed")
            model,loader,estimate,settings=self._task_models[name]
            if self._task_actor is None:
                self._task_actor=InferenceActor(self._batching_config)
            actor=self._task_actor
        if loader is None and not hasattr(model,"model"):
            if task.value == "vision-language-generation":
                from mlx_one.vision.loading import load_vlm_model

                loader=load_vlm_model
            else:
                from mlx_one.retrieval.loading import load_retrieval_model

                loader=load_retrieval_model
        lease=self.pool.acquire(name,lambda:loader(model,**settings) if loader else model,
            estimated_bytes=estimate,measure=lambda b:array_nbytes(b.model.parameters()))
        try:
            binding=get_runtime_registration(lease.runtime.value.architecture)
            tasks,_=binding.specifications()
            if task not in {s.task for s in tasks}:
                raise ValueError("task has no registered native execution contract")
            adapter=binding.adapter(task)(lease.runtime.value)
            prepared=adapter.prepare(task,payload)
        except BaseException:
            lease.release()
            raise
        return actor,adapter,prepared,lease

    def submit_task(self,name,task,payload,*,cancel=None,deadline=None):
        actor,adapter,prepared,lease=self._prepare_task(name,task,payload)
        return actor.submit_task(adapter,prepared,lease,cancel=cancel,deadline=deadline)

    def stream_task(self,name,task,payload,*,cancel=None):
        actor,adapter,prepared,lease=self._prepare_task(name,task,payload)
        def release():
            try:
                adapter.dispose_prepared(prepared)
            finally:
                lease.release()
        try:
            if not callable(getattr(adapter,"stream",None)):
                raise ValueError("task has no native stream contract")
            job=actor._enqueue(lambda event:adapter.stream(prepared,event),
                cancel=cancel,on_cancel=release)
        except BaseException:
            release()
            raise
        yield from actor.read_sync(job)

    def run_task(self,name,task,payload,**settings):
        return self.submit_task(name,task,payload,**settings).result()

    def embed(self,name,texts,**settings):
        from mlx_one.engine.contracts import TaskKind

        return self.run_task(name,TaskKind.EMBEDDING,{'texts':texts,**settings}).output

    def rerank(self,name,query,documents,**settings):
        from mlx_one.engine.contracts import TaskKind

        return self.run_task(name,TaskKind.RERANKING,
            {'query':query,'documents':documents,**settings}).output

    def unload(self, model, *, timeout=None):
        self.pool.unload(model, timeout=timeout)

    def close(self):
        with self._lock:
            self._closed = True
        if self._task_actor is not None:
            self._task_actor.close()
        self.pool.close()
