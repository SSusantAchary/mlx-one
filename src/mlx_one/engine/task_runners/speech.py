"""Native speech task bridges and transactional encoder/decoder primitives.

Transcript policy remains in audio.service (language, timestamps, EOS and resource
limits). Offline bridges are singleton; component batching is separately exposed
for compatible encoder shapes/decoder offsets and is never inferred from queue rows.
"""
from __future__ import annotations

import math
from array import array
from dataclasses import dataclass
from pathlib import Path

from mlx_one.engine.cache_errors import CacheCapacityError
from mlx_one.engine.contracts import TaskKind
from mlx_one.engine.task_runners.base import NativeTaskAdapter
from mlx_one.engine.tasks import PreparedTask, TaskFailure


@dataclass
class SpeechDecoderState:
    encoded: object
    cache: tuple
    closed: bool = False

    def close(self):
        if self.closed:
            return
        self.closed=True
        error=None
        for component in self.cache:
            try:
                component.reset()
            except BaseException as exc:
                error=error or exc
        self.encoded=None
        if error is not None:
            raise error


class EncoderDecoderSpeechRunner:
    """Ready decoder rows share evaluated forwards; cross features remain private."""
    def __init__(self,model,family):
        if family not in {'whisper','moonshine'}:
            raise ValueError('encoder-decoder runner requires Whisper or Moonshine')
        self.model,self.family=model,family
        self.forward_batches=[]

    def encode(self,inputs):
        import mlx.core as mx

        if not inputs or len({tuple(x.shape) for x in inputs})!=1:
            raise ValueError('encoder batching requires equal prepared shapes')
        encoded=self.model.encode(mx.concatenate(inputs,axis=0))
        if self.family=='whisper':
            encoded=encoded[0]
        mx.eval(encoded)
        if encoded.shape[0]!=len(inputs):
            raise ValueError('encoder row mapping mismatch')
        return tuple(SpeechDecoderState(encoded[i:i+1],self.model.make_cache())
                     for i in range(len(inputs)))

    def decode(self,states,ids,*,cancelled=None):
        import mlx.core as mx

        from mlx_one.core.cache import EncoderDecoderKVCache

        if (not states or len({id(s) for s in states})!=len(states)
                or ids.shape[0]!=len(states) or ids.ndim!=2
                or any(s.closed for s in states)
                or len({tuple(s.encoded.shape[1:]) for s in states})!=1
                or len({tuple(c.offset for c in s.cache) for s in states})!=1
                or any(len(s.cache)!=len(states[0].cache) for s in states)):
            raise ValueError('speech decoder layout or request mapping mismatch')
        packed=tuple(EncoderDecoderKVCache.merge(tuple(s.cache[i] for s in states))
                     for i in range(len(states[0].cache)))
        staged=[]
        try:
            encoded=mx.concatenate([s.encoded for s in states],axis=0)
            output=(self.model(None,ids,encoder_outputs=encoded,cache=packed,use_cache=True)
                    if self.family=='whisper' else self.model.decode(ids,encoded,cache=packed))
            mx.eval(output.logits)
            if output.logits.shape[:2]!=ids.shape:
                raise ValueError('speech logits/result mapping mismatch')
            staged=[tuple(c.batch_select((i,)) for c in packed) for i in range(len(states))]
            mx.eval(*(t for row in staged for c in row for t in
                      (c.self_attention.keys,c.self_attention.values,c.cross_keys,c.cross_values)
                      if t is not None))
            self.forward_batches.append(len(states))
            # No ownership is published until every row has evaluated and validated.
            for i,state in enumerate(states):
                if cancelled is not None and cancelled(i):
                    continue
                for original,replacement in zip(state.cache,staged[i],strict=True):
                    original.__dict__,replacement.__dict__=replacement.__dict__,original.__dict__
            return output.logits
        finally:
            for row in (packed,*staged):
                for cache in row:
                    cache.reset()


class AudioTaskAdapter(NativeTaskAdapter):
    supported_tasks=(TaskKind.TRANSCRIPTION,)

    def __init__(self,bundle):
        if bundle.model.config.model_type!=self.family:
            raise ValueError('audio adapter/checkpoint mismatch')
        self.bundle=bundle

    def prepare(self,task,payload):
        task=TaskKind(task)
        if task not in self.supported_tasks:
            raise ValueError('audio task is unavailable for this checkpoint')
        rate=payload.get('sample_rate')
        if isinstance(rate,bool) or rate!=16000:
            raise ValueError('task waveforms require explicit sample_rate=16000')
        source=payload['audio']
        if isinstance(source,(str,Path)):
            raise ValueError('task admission requires decoded bounded mono PCM, not encoded files')
        # Copy host samples before queueing; caller mutations cannot change admitted work.
        if getattr(source,'ndim',1)!=1:
            raise ValueError('audio must be mono')
        if len(source)>180*rate:
            raise CacheCapacityError('audio exceeds the 180-second task bound',retryable=False)
        samples=array('f',source)
        if any(not math.isfinite(x) for x in samples):
            raise ValueError('audio samples must be finite')
        data=dict(payload,audio=samples)
        if task==TaskKind.ALIGNMENT and (not isinstance(data.get('text'),str)
                                       or not data.get('language')):
            raise ValueError('known-text alignment requires supplied text and language')
        frames=max(1,(len(samples)+159)//160)
        # Analytical upper bound includes dense encoder attention, features and PCM.
        heads=getattr(self.bundle.model.config,'encoder_attention_heads',16)
        bound=frames*frames*heads*4*5+len(samples)*16
        return PreparedTask(task,data,(max(1,(len(samples)+15999)//16000),),
                            (self.family,task.value),bound)

    def run_steps(self,prepared,controls):
        from mlx_one.audio.qwen import align_audio
        from mlx_one.audio.service import detect_audio_language, transcribe

        outputs=[]
        for p,control in zip(prepared,controls,strict=True):
            data=dict(p.payload)
            audio=data.pop('audio')
            try:
                if p.task==TaskKind.ALIGNMENT:
                    output=align_audio(self.bundle,audio,cancelled=control.cancelled,**data)
                elif p.task == TaskKind.LANGUAGE_DETECTION:
                    output=detect_audio_language(self.bundle,audio,
                        cancelled=control.cancelled,**data)
                else:
                    output=transcribe(self.bundle,audio,cancelled=control.cancelled,**data)
            except Exception as error:
                output=TaskFailure(error)
            outputs.append(output)
            yield None
        return tuple(outputs)

    def run_batch(self,prepared):
        from threading import Event

        from mlx_one.engine.tasks import TaskControl

        steps=self.run_steps(prepared,tuple(TaskControl(Event(),None) for _ in prepared))
        while True:
            try:
                next(steps)
            except StopIteration as result:
                return result.value


class WhisperAdapter(AudioTaskAdapter):
    family='whisper'
    supported_tasks=(TaskKind.TRANSCRIPTION,TaskKind.LANGUAGE_DETECTION)


class MoonshineAdapter(AudioTaskAdapter):
    family='moonshine'


class Qwen3ASRAdapter(AudioTaskAdapter):
    family='qwen3_asr'
    supported_tasks=(TaskKind.TRANSCRIPTION,TaskKind.LANGUAGE_DETECTION)


class Qwen3AlignmentAdapter(AudioTaskAdapter):
    family='qwen3_forced_aligner'
    supported_tasks=(TaskKind.ALIGNMENT,)
