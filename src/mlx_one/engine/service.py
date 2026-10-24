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
        self._sessions = set()
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
        from mlx_one.engine.adapter_registry import get_runtime_registration
        from mlx_one.engine.contracts import canonical_task

        task=canonical_task(task)
        actor,lease=self._task_lease(name,task)
        try:
            binding=get_runtime_registration(getattr(lease.runtime.value,"architecture",None)
                or lease.runtime.value.model.config.model_type)
            tasks,_=binding.specifications()
            if task not in {s.task for s in tasks}:
                raise ValueError("task has no registered native execution contract")
            adapter=binding.adapter(task)(lease.runtime.value)
            prepared=adapter.prepare(task,payload)
        except BaseException:
            lease.release()
            raise
        return actor,adapter,prepared,lease

    def _task_lease(self,name,task):
        from mlx_one.engine.actor import InferenceActor
        from mlx_one.engine.cache import array_nbytes
        from mlx_one.engine.contracts import TaskKind, canonical_task

        task = canonical_task(task)
        with self._lock:
            if self._closed:
                raise RuntimeError("inference service is closed")
            model,loader,estimate,settings=self._task_models[name]
            if self._task_actor is None:
                self._task_actor=InferenceActor(self._batching_config)
            actor=self._task_actor
        if loader is None and not hasattr(model,"model"):
            if task==TaskKind.VISION_LANGUAGE:
                from mlx_one.vision.loading import load_vlm_model

                loader=load_vlm_model
            elif task in {TaskKind.TRANSCRIPTION,TaskKind.ALIGNMENT,TaskKind.LANGUAGE_DETECTION,
                          TaskKind.SPEECH_GENERATION,TaskKind.SPEECH_TO_SPEECH}:
                from mlx_one.audio.loading import load_audio_model

                loader=load_audio_model
                settings=dict(settings,task={TaskKind.ALIGNMENT:"align",
                    TaskKind.SPEECH_GENERATION:"synthesize",
                    TaskKind.SPEECH_TO_SPEECH:"speech-to-speech"}.get(task,"transcribe"))
            elif task in {TaskKind.IMAGE_SEGMENTATION,TaskKind.MASK_GENERATION,
                          TaskKind.VIDEO_SEGMENTATION}:
                from mlx_one.segmentation.loading import load_segmentation_model

                loader=load_segmentation_model
                settings=dict(settings,task=settings.get("task") or (
                    "video" if task==TaskKind.VIDEO_SEGMENTATION else
                    "interactive" if task==TaskKind.MASK_GENERATION else "concept"))
            else:
                from mlx_one.retrieval.loading import load_retrieval_model

                loader=load_retrieval_model
        lease=self.pool.acquire(name,lambda:loader(model,**settings) if loader else model,
            estimated_bytes=estimate,measure=lambda b:array_nbytes(b.model.parameters()) + (
                array_nbytes(b.codec.parameters()) + array_nbytes(getattr(b.codec, '_window', None))
                if getattr(b, 'codec', None) is not None else 0))
        return actor,lease

    def open_audio_session(self,name,*,sample_rate,language=None,options=None,
                           cancel=None,deadline=None):
        from mlx_one.audio.streaming import open_audio_stream
        from mlx_one.engine.contracts import TaskKind
        from mlx_one.engine.sessions import DeviceSession

        actor,lease=self._task_lease(name,TaskKind.TRANSCRIPTION)
        event=cancel or threading.Event()
        session=DeviceSession(lease,actor.device_executor,lambda:open_audio_stream(lease.runtime.value,
            sample_rate=sample_rate,language=language,options=options,cancelled=event.is_set),
            methods=("push","finish","reset"),cancel=event,deadline=deadline)
        with self._lock:
            if self._closed:
                session.close()
                raise RuntimeError("inference service is closed")
            self._sessions.add(session)
            session.on_close=lambda:self._sessions.discard(session)
        return session

    def open_video_session(self,name,*,memory_budget_bytes=24*1024**3,
                           cancel=None,deadline=None):
        from mlx_one.engine.contracts import TaskKind
        from mlx_one.engine.sessions import DeviceSession

        if memory_budget_bytes<1:
            raise ValueError("video state requires a positive byte budget")
        actor,lease=self._task_lease(name,TaskKind.VIDEO_SEGMENTATION)
        session=DeviceSession(lease,actor.device_executor,lambda:lease.runtime.value.video_session(
            streaming=True,memory_budget_bytes=memory_budget_bytes),
            methods=("add_prompt","remove_prompt","remove_object","correct_object",
                     "process_frame","reset"),cancel=cancel,deadline=deadline)
        with self._lock:
            if self._closed:
                session.close()
                raise RuntimeError("inference service is closed")
            self._sessions.add(session)
            session.on_close=lambda:self._sessions.discard(session)
        return session

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

    def task_capabilities(self,name,task):
        from dataclasses import asdict

        from mlx_one.engine.adapter_registry import get_runtime_registration
        from mlx_one.engine.contracts import canonical_task
        from mlx_one.engine.reporting import tensor_dtypes

        task=canonical_task(task)
        _,lease=self._task_lease(name,task)
        try:
            bundle=lease.runtime.value
            architecture=getattr(bundle,"architecture",None) or bundle.model.config.model_type
            binding=get_runtime_registration(architecture)
            specs,_=binding.specifications()
            available=task in {s.task for s in specs} and not binding.component_only
            descriptor=binding.adapter(task)(bundle).describe(bundle) if available else None
            return {"architecture":architecture,"task":task.value,
                "availability":{"implemented":available,"opt_in":available,
                    "reason":binding.reason},"adapter":asdict(descriptor) if descriptor else None,
                "actual_weight_dtypes":tensor_dtypes(bundle.model.parameters()),
                "model_epoch":lease.runtime.epoch,
                "validation":{"checkpoint_validated":False,"hardware_qualified":False},
                "qualification":{"performance_qualified":False,
                    "reason":"No exact task/profile evidence selected; defaults remain unchanged"}}
        finally:
            lease.release()

    def unload(self, model, *, timeout=None):
        self.pool.unload(model, timeout=timeout)

    def close(self):
        with self._lock:
            self._closed = True
        error=None
        releases=[session.close for session in tuple(self._sessions)]
        if self._task_actor is not None:
            releases.append(self._task_actor.close)
        releases.append(self.pool.close)
        for release in releases:
            try:
                release()
            except BaseException as exc:
                error=error or exc
        if error is not None:
            raise error
