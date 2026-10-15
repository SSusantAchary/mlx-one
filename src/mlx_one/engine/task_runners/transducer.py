"""Ready TDT predictor/joint rows; time advances independently from emitted text."""
from __future__ import annotations

import resource
import time
from dataclasses import dataclass, field

from mlx_one.engine.task_runners.speech import AudioTaskAdapter
from mlx_one.engine.tasks import TaskFailure


@dataclass
class TDTState:
    length: int
    blank: int
    limit: int
    frame: int = 0
    previous: int | None = None
    predictor: object = None
    decoder: object = None
    tokens: list = field(default_factory=list)
    durations: list = field(default_factory=list)
    finished: bool = False

    def __post_init__(self):
        if self.length<0 or self.limit<1:
            raise ValueError('invalid TDT length or emission budget')
        self.previous=self.blank
        self.tokens=[self.blank]
        self.durations=[0]
        self.finished=self.length==0

    def transition(self,token,duration,*,eos=None):
        if self.finished or token<0 or duration<0:
            raise ValueError('invalid TDT transition')
        if len(self.tokens)>self.limit:
            from mlx_one.audio.moonshine import AudioResourceLimitError

            raise AudioResourceLimitError('TDT emission budget exhausted; refusing truncation')
        duration=1 if token==self.blank and duration==0 else duration
        self.tokens.append(token)
        self.durations.append(duration)
        self.previous=token
        self.frame+=duration
        self.finished=self.frame>=self.length or token==eos

    def close(self):
        self.predictor=self.decoder=None
        self.finished=True


class TransducerRunner:
    def __init__(self,model):
        self.model=model
        self.forward_batches=[]

    def step(self,encoded,states,*,cancelled=None,eos=None):
        import mlx.core as mx

        ready=[i for i,s in enumerate(states) if not s.finished and not (
            cancelled is not None and cancelled(i))]
        if not ready:
            return
        staged={}
        predict=[i for i in ready if states[i].decoder is None
                 or states[i].previous!=states[i].blank]
        if predict:
            config=self.model.config
            def initial(s):
                if s.predictor is not None:
                    return s.predictor
                value=mx.zeros((config.num_decoder_layers,1,config.decoder_hidden_size))
                return value,value
            prior=tuple(initial(states[i]) for i in predict)
            packed=tuple(mx.concatenate([p[axis] for p in prior],axis=1) for axis in (0,1))
            output,next_state=self.model.decoder(mx.array([states[i].previous for i in predict]),
                                                 packed)
            mx.eval(output,*next_state)
            for row,i in enumerate(predict):
                staged[i]=(output[row:row+1],tuple(x[:,row:row+1] for x in next_state))
        decoders=mx.concatenate([staged[i][0] if i in staged else states[i].decoder
                                for i in ready],axis=0)
        frames=mx.concatenate([encoded[i:i+1,states[i].frame] for i in ready],axis=0)
        logits=self.model.joint(frames,decoders)
        mx.eval(logits)
        if logits.shape[0]!=len(ready):
            raise ValueError('TDT joint row mapping mismatch')
        config=self.model.config
        tokens=mx.argmax(logits[:,:config.vocab_size],axis=-1).tolist()
        durations=mx.argmax(logits[:,config.vocab_size:],axis=-1).tolist()
        self.forward_batches.append(len(ready))
        for row,i in enumerate(ready):
            if cancelled is not None and cancelled(i):
                continue
            states[i].transition(int(tokens[row]),int(durations[row]),eos=eos)
            if i in staged:
                states[i].decoder,states[i].predictor=staged[i]


class ParakeetTDTAdapter(AudioTaskAdapter):
    family='parakeet_tdt'

    def prepare(self,task,payload):
        from mlx_one.audio.schemas import ASRDecodeOptions

        if payload.get('task','transcribe')!='transcribe' or payload.get('word_timestamps',False):
            raise ValueError('Parakeet provides offline token timestamps, not word timestamps')
        options=payload.get('options')
        if options is not None and type(options) is not ASRDecodeOptions:
            raise ValueError('Parakeet requires ASRDecodeOptions')
        language=payload.get('language')
        if (self.bundle.variant=='v2' and language not in (None,'en','English','english')
                or self.bundle.variant=='v3' and language is not None):
            raise ValueError('unsupported Parakeet language conditioning')
        return super().prepare(task,payload)

    def run_steps(self,prepared,controls):
        import mlx.core as mx

        from mlx_one.audio.moonshine import BUDGET, AudioResourceLimitError
        from mlx_one.audio.parakeet import timestamp_tokens
        from mlx_one.audio.schemas import SpeechTranscript
        from mlx_one.models.audio.parakeet.processing import preprocess

        started=time.perf_counter()
        features=[]
        counts=[]
        states=[]
        outputs=[None]*len(prepared)
        try:
            for p in prepared:
                samples=p.payload['audio']
                if samples:
                    feature,count=preprocess(mx.array(samples),self.bundle.processor)
                else:
                    feature,count=mx.zeros((1,1,self.bundle.model.config.num_mel_bins)),0
                features.append(feature)
                counts.append(count)
            width=max(x.shape[1] for x in features)
            packed=mx.concatenate([mx.pad(x,((0,0),(0,width-x.shape[1]),(0,0)))
                                   for x in features],axis=0)
            def check():
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>BUDGET:
                    raise AudioResourceLimitError('audio process memory ceiling exceeded')
            encoded,lengths=self.bundle.model.encode(packed,mx.array(counts),check=check)
            mx.eval(encoded,lengths)
            yield None
            config=self.bundle.model.config
            for i,p in enumerate(prepared):
                options=p.payload.get('options')
                length=int(lengths[i].item()) if counts[i] else 0
                states.append(TDTState(length,config.blank_token_id,
                    options.max_tokens if options and options.max_tokens else
                    max(1,config.max_symbols_per_step*length)))
            runner=TransducerRunner(self.bundle.model)
            while any(not s.finished for s in states):
                check()
                for i,state in enumerate(states):
                    if controls[i].cancelled():
                        state.close()
                    elif not state.finished and len(state.tokens)>state.limit:
                        outputs[i]=TaskFailure(AudioResourceLimitError(
                            'TDT emission budget exhausted; refusing truncation'))
                        state.close()
                runner.step(encoded,states,cancelled=lambda i:controls[i].cancelled(),
                            eos=self.bundle.generation_config.get('eos_token_id'))
                yield None
            for i,(p,state) in enumerate(zip(prepared,states,strict=True)):
                if outputs[i] is not None:
                    continue
                spans=timestamp_tokens(self.bundle.tokenizer,state.tokens,state.durations,
                                       blank=config.blank_token_id,pad=config.pad_token_id)
                outputs[i]=SpeechTranscript(self.bundle.tokenizer.decode(state.tokens,
                    skip_special_tokens=True),len(p.payload['audio'])/16000,str(self.bundle.path),
                    self.bundle.revision,16000,tokens=tuple(state.tokens),timestamps=spans,
                    language=('en' if self.bundle.variant=='v2'
                              and p.payload.get('language') else None),
                    timings={'wall_seconds':time.perf_counter()-started},
                    provenance={'model_type':self.family,'streaming':False,'weight_dtype':'float32',
                        'timestamp_tick_seconds':.08,'qualification':'WIP',
                        'joint_forward_batches':tuple(runner.forward_batches)})
            return tuple(outputs)
        finally:
            for state in states:
                state.close()
