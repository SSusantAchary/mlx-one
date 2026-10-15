"""One offline native task configuration per supervised process."""
from __future__ import annotations

import argparse
import dataclasses
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from mlx_one.native_qualification import write
from mlx_one.runtime_baseline import file_digest
from mlx_one.task_qualification import read_inputs, read_profile, task_gates


def serializable(value):
    if dataclasses.is_dataclass(value):
        return {f.name:serializable(getattr(value,f.name)) for f in dataclasses.fields(value)
                if f.name not in {'timings','provenance','execution_identity'}}
    if isinstance(value,dict):
        return {k:serializable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [serializable(v) for v in value]
    if hasattr(value,'tolist'):
        return value.tolist()
    return value


def compare_result(left,right,*,atol,rtol):
    import numpy as np

    if isinstance(left,dict) and isinstance(right,dict):
        return left.keys()==right.keys() and all(compare_result(left[k],right[k],
            atol=atol,rtol=rtol) for k in left)
    if isinstance(left,list) and isinstance(right,list):
        return len(left)==len(right) and all(compare_result(a,b,atol=atol,rtol=rtol)
                                             for a,b in zip(left,right,strict=True))
    if isinstance(left,(float,int)) and isinstance(right,(float,int)):
        if isinstance(left,(int,bool)) and isinstance(right,(int,bool)):
            return left==right
        return bool(np.isclose(left,right,atol=atol,rtol=rtol,equal_nan=False))
    return left==right


def task_quality(task,outputs,cases,limits,manifest_sha256):
    """Ground truth is required; independent native agreement alone is not quality."""
    if any('expected' not in case for case in cases):
        return {'passed':None,'reason':'ground truth task anchors not supplied',
                'ground_truth_sha256':None}
    checks=[]
    if task=='transcription':
        from mlx_one.audio.evaluation import word_error_rate

        for output,case in zip(outputs,cases,strict=True):
            expected=case['expected']['text']
            score=word_error_rate(expected,output['text'])
            limit=limits.get('max_wer')
            checks.append(None if limit is None else score<=limit)
    else:
        atol=limits.get('quality_atol')
        rtol=limits.get('quality_rtol')
        if task=='alignment' and atol is not None:
            atol=min(atol,.08)
            rtol=0.
        for output,case in zip(outputs,cases,strict=True):
            expected=case['expected']
            checks.append(None if atol is None or rtol is None else compare_result(
                {k:output[k] for k in expected},expected,atol=atol,rtol=rtol))
    return {'passed':False if False in checks else None if None in checks else True,
            'checks':checks,'ground_truth_sha256':manifest_sha256,'coverage_passed':None,
            'scope':'declared task anchors only; no unseen dataset generalization'}


def _arrays(value):
    if isinstance(value,(tuple,list)):
        return [a for v in value for a in _arrays(v)]
    if isinstance(value,dict):
        return [a for v in value.values() for a in _arrays(v)]
    if hasattr(value,'dtype') and hasattr(value,'shape'):
        return [value]
    return [a for name in ('logits','embeddings','last_hidden_state','encoder_last_hidden_state')
            if (a:=getattr(value,name,None)) is not None and hasattr(a,'dtype')]


@contextmanager
def observe_calls(bundle):
    """Measure actual native root/encoder/decoder/joint calls, not queued rows."""
    import mlx.core as mx

    from mlx_one.engine.transactions import component_tensors

    class CallTrace(list):
        cohort="oracle"

    traces=CallTrace()
    lock=threading.Lock()
    patches=[]
    targets=[(type(bundle.model),name) for name in ('__call__','encode','decode')
             if callable(getattr(type(bundle.model),name,None))]
    for component in ('joint','mtp'):
        if hasattr(bundle.model,component):
            targets.append((type(getattr(bundle.model,component)),'__call__'))
    for cls,name in targets:
        original=getattr(cls,name)
        def wrapper(self,*args,_original=original,_stage=name,**kwargs):
            started=time.perf_counter()
            output=_original(self,*args,**kwargs)
            mx.eval(*_arrays(output))
            cache=kwargs.get('cache') or getattr(output,'cache',()) or ()
            layers=[]
            for layer,component in enumerate(cache):
                tensors=list(component_tensors(component))
                if hasattr(component,'self_attention'):
                    tensors+=list(component_tensors(component.self_attention))
                    tensors += [v for v in (component.cross_keys,component.cross_values)
                                if v is not None]
                layers.append({'layer_id':layer,'position':component.offset,
                    'component':type(component).__name__,
                    'actual_dtypes':sorted({str(t.dtype).rsplit('.',1)[-1] for t in tensors})})
            first=next((v for v in args if hasattr(v,'shape') and len(v.shape)),None)
            with lock:
                if len(traces)<8192:
                    traces.append({'cohort':traces.cohort,'stage':_stage,'batch_size':int(first.shape[0])
                        if first is not None else None,'output_dtypes':sorted({
                            str(a.dtype).rsplit('.',1)[-1]
                            for a in _arrays(output)}),'input_shape':list(first.shape)
                        if first is not None else None,'duration_ms':
                        (time.perf_counter()-started)*1000,'cache_components':layers})
            return output
        setattr(cls,name,wrapper)
        patches.append((cls,name,original))
    try:
        yield traces
    finally:
        for cls,name,original in reversed(patches):
            setattr(cls,name,original)


def load_bundle(profile):
    from mlx_one.engine.contracts import TaskKind, canonical_task

    task=canonical_task(profile.task)
    settings={'revision':profile.revision,'offline':True}
    if task==TaskKind.TEXT_GENERATION:
        from mlx_one.text.loading import load_text_model

        bundle=load_text_model(profile.checkpoint_path,**settings)
    elif task==TaskKind.VISION_LANGUAGE:
        from mlx_one.vision.loading import load_vlm_model

        bundle=load_vlm_model(profile.checkpoint_path,**settings)
    elif task in {TaskKind.TRANSCRIPTION,TaskKind.ALIGNMENT}:
        from mlx_one.audio.loading import load_audio_model

        bundle=load_audio_model(profile.checkpoint_path,
                               task='align' if task==TaskKind.ALIGNMENT else 'transcribe',
                               **settings)
    elif task in {TaskKind.IMAGE_SEGMENTATION,TaskKind.MASK_GENERATION,TaskKind.VIDEO_SEGMENTATION}:
        from mlx_one.segmentation.loading import load_segmentation_model

        bundle=load_segmentation_model(profile.checkpoint_path,
            task=profile.settings.get('segmentation_task') or ('video'
                if task==TaskKind.VIDEO_SEGMENTATION else 'interactive'
                if task==TaskKind.MASK_GENERATION else 'concept'),**settings)
    else:
        from mlx_one.retrieval.loading import load_retrieval_model

        bundle=load_retrieval_model(profile.checkpoint_path,**settings)
    observed=getattr(bundle,'architecture',None) or bundle.model.config.model_type
    if observed!=profile.model_type:
        raise ValueError('checkpoint architecture substitution is forbidden')
    if profile.execution_dtype is not None:
        if task in {TaskKind.TRANSCRIPTION,TaskKind.ALIGNMENT,TaskKind.IMAGE_SEGMENTATION,
                    TaskKind.MASK_GENERATION,TaskKind.VIDEO_SEGMENTATION}:
            raise ValueError('speech/segmentation reference precision must remain native')
        from mlx_one.engine.modes import configure_execution_dtype

        configure_execution_dtype(bundle,profile.execution_dtype)
    return bundle


def payload_for(case,root):
    import numpy as np

    payload=dict(case['payload'])
    assets={a['path'] for a in case.get('assets',[])}
    def asset(name):
        if name not in assets:
            raise ValueError('every input media path requires a declared checksum')
        return str((root/name).resolve())
    if 'images' in payload:
        payload['images']=[asset(p) for p in payload['images']]
    if 'image' in payload:
        payload['image']=asset(payload['image'])
    if 'audio_file' in payload:
        file=asset(payload.pop('audio_file'))
        if not file.endswith('.npy'):
            raise ValueError(
                'task audio fixtures require decoded mono .npy PCM; no implicit resampling')
        payload['audio']=np.load(file,allow_pickle=False)
    return payload


def run(profile,output):
    import mlx.core as mx
    import numpy as np

    from mlx_one.engine.batching import BatchingConfig
    from mlx_one.engine.pressure import sample_telemetry
    from mlx_one.engine.service import InferenceService
    from mlx_one.inference_baseline import source_provenance

    data,root=read_inputs(profile)
    cases=data['cases']
    bundle=load_bundle(profile)
    batching=dict(profile.settings.get('batching',{}))
    if batching.get('max_sequences',profile.slots)!=profile.slots:
        raise ValueError('profile slots conflict with configured task rows')
    batching.setdefault('max_sequences',profile.slots)
    batching.setdefault('queue_size',max(len(cases),8))
    config=BatchingConfig(**batching)
    service=InferenceService(batching_config=config)
    if profile.task=='text-generation':
        from mlx_one.engine.public import Engine

        settings=dict(profile.settings.get('engine',{}))
        if settings.get('inference_mode') not in {'reference_dense','experimental_block',
                'experimental_dense_batch','experimental_hybrid_batch','paged'}:
            raise ValueError('text benchmark requires an explicit resolved mode; auto is forbidden')
        target=Engine(bundle,batching_config=config,execution_dtype=profile.execution_dtype,
                      **settings)
    else:
        target=None
        service.register_task('checkpoint',bundle)
    def execute(case):
        started=time.perf_counter()
        payload=payload_for(case,root)
        if target is not None:
            from mlx_one.text.schemas import TextGenerationOptions

            events=list(target.stream(payload['prompt'],options=TextGenerationOptions(
                **payload.get('generation',{}))))
            result={'tokens':[e.token_id for e in events if e.token_id is not None],
                    'text':''.join(e.text for e in events),'finish_reason':events[-1].finish_reason}
        elif profile.task=='video-segmentation':
            from PIL import Image

            with service.open_video_session('checkpoint',memory_budget_bytes=
                    profile.settings.get('video_memory_budget_bytes',24*1024**3)) as session:
                session.call('add_prompt',payload['concept'])
                frames=payload['frames']
                if len(frames)>64:
                    raise ValueError('video qualification retains at most 64 frame results')
                result=[]
                for i,name in enumerate(frames):
                    if name not in {a['path'] for a in case.get('assets',[])}:
                        raise ValueError('frame checksum is missing')
                    with Image.open(root/name) as frame:
                        result.append(session.call('process_frame',frame,frame_index=i))
        else:
            result=service.run_task('checkpoint',profile.task,payload).output
        return serializable(result),{'request_id':case['request_id'],
                                    'response_ms':(time.perf_counter()-started)*1000}
    outputs=[]
    timings=[]
    references=[]
    parity=[]
    try:
        with observe_calls(bundle) as traces:
            references=[execute(case)[0] for case in cases]
            # One explicitly separate warm-up before measured repetitions.
            traces.cohort='warmup'
            for case in cases:
                execute(case)
            traces.cohort='measured'
            measured_start=time.perf_counter()
            for repetition in range(profile.repetitions):
                with ThreadPoolExecutor(max_workers=profile.slots) as executor:
                    results=list(executor.map(execute,cases))
                outputs.append([r[0] for r in results])
                timings.extend([{**r[1],'repetition':repetition} for r in results])
                atol=profile.limits.get('parity_atol',1e-4)
                rtol=profile.limits.get('parity_rtol',1e-4)
                parity.extend(compare_result(actual,expected,atol=atol,rtol=rtol)
                    for (actual,_),expected in zip(results,references,strict=True))
            mx.synchronize()
            seconds=time.perf_counter()-measured_start
        telemetry=sample_telemetry()
        stats=service.pool.stats()
        owner=target._cache_manager.stats() if target is not None else None
        measurement={'backend':'mlx-metal','metal_available':bool(mx.metal.is_available()),
            'rss_bytes':telemetry.rss_bytes,'mlx_active_bytes':int(mx.get_active_memory()),
            'mlx_cached_bytes':int(mx.get_cache_memory()),'mlx_peak_bytes':int(mx.get_peak_memory()),
            'active_handles':owner.active_handles if owner else 0,
            'reservations_bytes':owner.reserved_bytes if owner else 0,
            'active_leases':stats['active_leases'],
            'response_p95_ms':float(np.percentile([t['response_ms'] for t in timings],95)),
            'throughput_per_second':len(timings)/seconds,'throughput_unit':'completed tasks/s',
            'model_token_gaps_ms':None,'sse_timing_ms':None,
            'timing_scope':'native task completion; not HTTP service or model decode throughput',
            'source':source_provenance(Path.cwd()),'profile':dataclasses.asdict(profile),
            'hardware':json.loads((output/'manifest.json').read_text()).get('hardware'),
            'reference':None,'forward_count':len(traces),
            'actual_max_forward_batch':max((t['batch_size'] or 1 for t in traces
                if t['cohort']=='measured'),default=0),
            'trace_truncated':len(traces)>=8192,'resolved_execution_settings':
                dataclasses.asdict(config)}
        if profile.reference_run:
            from mlx_one.inference_qualification import validate_integrity

            reference=Path(profile.reference_run)
            validate_integrity(reference,required=('measurement.json','manifest.json'))
            measurement['reference']=json.loads((reference/'measurement.json').read_text())
        proof={'passed':all(parity),'comparisons':parity,
               'oracle':'independent native task execution'}
        quality=task_quality(profile.task,outputs[-1],cases,profile.limits,profile.inputs_sha256)
        gates=task_gates(profile,proof,measurement,quality,reference=measurement['reference'])
        for name,value in (('measurement.json',measurement),('task-parity.json',proof),
            ('task-quality.json',quality),('task-gates.json',gates),('outputs.json',outputs),
            ('oracle.json',references),('timings.json',timings),('forward-traces.json',traces)):
            write(output/name,value)
        files={str(p.relative_to(bundle.path)):file_digest(p) for p in sorted(
            bundle.path.rglob('*')) if p.is_file()
               and p.suffix in {'.json','.safetensors','.jinja'}}
        write(output/'checkpoint.json',{'model_id':profile.model_id,'revision':profile.revision,
                                       'files':files})
        from importlib.metadata import version

        from mlx_one.engine.adapters import ADAPTER_VERSION
        from mlx_one.engine.reporting import tensor_dtypes
        from mlx_one.registry import InferenceScope

        caches={dtype for trace in traces for c in trace['cache_components']
                for dtype in c['actual_dtypes']}
        config_digest=file_digest(bundle.path/'config.json')
        scope=InferenceScope(profile.model_id,profile.revision,'native-task-v1',config_digest,
            ','.join(tensor_dtypes(bundle.model.parameters())),
            ','.join(sorted(caches)) if caches else 'not_applicable',version('mlx'),
            measurement.get('hardware',{}).get('chip'),measurement['source']['git_revision'])
        unique={(c['layer_id'],c['component'],tuple(c['actual_dtypes']))
                for trace in traces for c in trace['cache_components']}
        components=[{'layer_id':layer,'component':kind,'actual_dtypes':list(dtypes)}
                    for layer,kind,dtypes in sorted(unique)]
        if not components:
            components=[{'component':'model-output','actual_dtypes':sorted({d for t in traces
                        for d in t['output_dtypes']})}]
        contract={'adapter_version':ADAPTER_VERSION,'execution_mode':scope.execution_path,
            'topology_digest':config_digest,'quantization_digest':file_digest(
                bundle.path/'config.json'),'components':components,
            'task_adapter_version':'native-task-adapter-v1'}
        write(output/'task-runtime.json',{'scope':dataclasses.asdict(scope),
            'execution_contract':contract,'checkpoint_validated':False,
            'qualification_promoted':False,'implementation_executed':True})
        if not proof['passed']:
            raise ValueError('native task parity failed; checkpoint expansion must stop')
    finally:
        if target is not None:
            target.close()
        service.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    output=Path(args.output)
    try:
        run(read_profile(args.profile),output)
    except Exception as error:
        write(output/'failure.json',{'status':'FAIL','reason':str(error),
                                    'type':type(error).__name__,'qualification_promoted':False})
        raise
    return 0


if __name__=='__main__':
    raise SystemExit(main())
