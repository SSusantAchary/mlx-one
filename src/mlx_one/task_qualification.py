"""Registry-derived task manifests and supervised native evidence, never promotion."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import socket
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from mlx_one.engine.adapter_registry import RUNTIME_CONTRACT_VERSION, runtime_coverage
from mlx_one.engine.contracts import canonical_task
from mlx_one.inference_baseline import source_provenance
from mlx_one.inference_qualification import validate_integrity, write_integrity
from mlx_one.native_qualification import preflight, supervised_command, write
from mlx_one.runtime_baseline import file_digest

TASK_PROFILE_VERSION='native-task-profile-v1'
TASK_EVIDENCE_VERSION='native-task-evidence-v1'
TASK_GATE_VERSION='native-task-gates-v1'
STATUSES={'PASS','FAIL','BLOCKED','SKIPPED','NOT_RUN'}
TASK_CORRECTNESS_CHECKS = frozenset({
    'parity', 'native_execution', 'checkpoint_scope', 'telemetry', 'cleanup', 'quality',
    'task_lifecycle', 'task_quality_coverage', 'five_repetitions',
})


@dataclass(frozen=True)
class TaskProfile:
    model_type: str
    task: str
    model_id: str
    revision: str
    checkpoint_path: str
    inputs_manifest: str
    inputs_sha256: str
    schema_version: str = TASK_PROFILE_VERSION
    slots: int = 1
    repetitions: int = 5
    warmups: int = 1
    timeout_seconds: float = 600
    execution_dtype: str | None = None
    settings: dict = field(default_factory=dict)
    limits: dict = field(default_factory=dict)
    reference_run: str | None = None
    validation_kind: str = 'checkpoint'

    def __post_init__(self):
        if self.schema_version!=TASK_PROFILE_VERSION or self.validation_kind not in {
                'checkpoint','synthetic-smoke'}:
            raise ValueError('current task profile schema/validation kind required')
        if (not re.fullmatch('[0-9a-f]{40,64}',self.revision)
                or not re.fullmatch('[0-9a-f]{64}',self.inputs_sha256)):
            raise ValueError('task profiles require pinned checkpoint and input checksums')
        if self.slots<1 or self.repetitions<1 or self.warmups!=1 or self.timeout_seconds<=0:
            raise ValueError('task profile requires positive bounds and one warm-up')
        if self.execution_dtype not in {None,'float16','bfloat16'}:
            raise ValueError('unsupported explicit execution precision')
        if any(isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v)
               or v<0 for v in self.limits.values()):
            raise ValueError("task limits must be finite nonnegative numbers")
        object.__setattr__(self,"task",canonical_task(self.task).value)
        implementation = self.settings.get('qwen3_implementation', 'canonical')
        if implementation not in {'canonical', 'upstream-candidate'} or (
                implementation == 'upstream-candidate'
                and (self.model_type != 'qwen3' or self.task != 'text-generation')):
            raise ValueError('Qwen3 implementation must be canonical or explicit text candidate')
        from mlx_one.engine.adapter_registry import get_runtime_registration

        binding=get_runtime_registration(self.model_type)
        declared={t.task for t in binding.specifications()[0]}
        if declared and canonical_task(self.task) not in declared:
            raise ValueError('task is not supported by selected architecture')
        if any('key' in k.lower() or 'credential' in k.lower() for k in self.settings):
            raise ValueError('credentials do not belong in qualification profiles')

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()


def read_profile(path):
    data=json.loads(Path(path).read_text())
    if not isinstance(data,dict):
        raise ValueError('task profile must be a JSON object')
    return TaskProfile(**data)


def task_inventory():
    return {'schema_version':TASK_PROFILE_VERSION,'runtime_contract':RUNTIME_CONTRACT_VERSION,
        'entries':list(runtime_coverage()),'qualification_promoted':False}


def read_inputs(profile):
    path=Path(profile.inputs_manifest).expanduser().resolve()
    if file_digest(path)!=profile.inputs_sha256:
        raise ValueError('task input manifest checksum mismatch')
    data=json.loads(path.read_text())
    if data.get('schema_version')!='native-task-inputs-v1' or not data.get('cases'):
        raise ValueError('task inputs require versioned nonempty cases')
    seen=set()
    for case in data['cases']:
        if not isinstance(case.get('request_id'),str) or case['request_id'] in seen:
            raise ValueError('task input request IDs must be unique')
        seen.add(case['request_id'])
        if not isinstance(case.get('payload'),dict):
            raise ValueError('task inputs require typed payload mappings')
        for asset in case.get('assets',[]):
            target=(path.parent/asset['path']).resolve()
            if not target.is_relative_to(path.parent) or file_digest(target)!=asset['sha256']:
                raise ValueError('task input asset path/checksum mismatch')
        if any(k in case for k in ('api_key','authorization','credentials','cache_namespace')):
            raise ValueError(
                'input manifests cannot grant namespace authority or contain credentials')
    if 'text_quality' in data:
        from mlx_one.quality_dataset import validate_text_quality

        if profile.task!='text-generation':
            raise ValueError('text quality protocol requires text-generation')
        validate_text_quality(data)
    return data,path.parent


def run_task_profile(profile,output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    source=source_provenance(Path.cwd())
    manifest={'schema_version':TASK_PROFILE_VERSION,'profile':asdict(profile),
              'profile_sha256':profile.digest,'source':source,'qualification_promoted':False}
    write(output/'manifest.json',manifest)
    try:
        binding=next(r for r in runtime_coverage() if r['model_type']==profile.model_type)
        if not binding['shared_service_integrated']:
            raise FileNotFoundError(binding['reason'])
        root=Path(profile.checkpoint_path).expanduser()
        if not root.is_dir() or not (root/'config.json').is_file():
            raise FileNotFoundError('pinned local checkpoint assets are missing')
        read_inputs(profile)
        report=preflight(output/'preflight')
        manifest['hardware']=report['hardware']
        write(output/'manifest.json',manifest)
        write(output/'preflight.json',report)
        write(output/'profile.json',asdict(profile))
        result=supervised_command([sys.executable,'-m','mlx_one.task_qualification_worker',
            '--profile',str((output/'profile.json').resolve()),'--output',str(output.resolve())],
            output/'worker',ceiling=report['process_ceiling_bytes'],timeout=profile.timeout_seconds,
            env={'PYTHONPATH':'src:.:tests','HF_HUB_OFFLINE':'1','MLX_ONE_OFFLINE':'1'})
        write(output/'supervision.json',result)
        if result['status']=='completed' and result['exit_code']==0:
            runtime=json.loads((output/'task-runtime.json').read_text())
            gates=json.loads((output/'task-gates.json').read_text())
            documents=('manifest.json','measurement.json','checkpoint.json','task-parity.json',
                'task-quality.json','task-gates.json','preflight.json','supervision.json',
                'task-runtime.json')
            write(output/'task-evidence.json',{'schema_version':'native-inference-qualification-v4',
                'task_evidence_version':TASK_EVIDENCE_VERSION,'backend':'mlx-one-native',
                'feature':'task:'+profile.task,'scope':runtime['scope'],
                'execution_contract':runtime['execution_contract'],
                'status':'passed' if gates['status']=='passed' else 'pending',
                'correctness_passed':False,'performance_qualified':False,
                'validation_kind':'native-hardware','qualification_promoted':False,
                'artifacts':[{'uri':name,'kind':'native-task-artifact',
                    'sha256':file_digest(output/name)} for name in documents]})

        if result['status']!='completed' or result['exit_code']!=0:
            write(output/'failure.json',{'status':'FAIL','reason':result,
                                        'qualification_promoted':False})
    except FileNotFoundError as error:
        write(output/'failure.json',{'status':'BLOCKED','reason':str(error)})
    except Exception as error:
        write(output/'failure.json',{'status':'FAIL','reason':str(error),'type':type(error).__name__})
        write_integrity(output)
        raise
    write_integrity(output)
    return output


def task_gates(profile,parity,measurement,quality,*,reference=None):
    """Missing service/quality data is pending; unrelated measurements cannot qualify."""
    limits=profile.limits
    checks={'parity':parity.get('passed'),
        'native_execution':measurement.get('backend')=='mlx-metal'
            and measurement.get('metal_available') is True,
        'checkpoint_scope':profile.validation_kind=='checkpoint',
        'telemetry':True if all(measurement.get(k) is not None for k in
            ('rss_bytes','mlx_active_bytes','mlx_cached_bytes','mlx_peak_bytes')) else None,
        'cleanup':measurement.get('active_handles')==0 and measurement.get('reservations_bytes')==0
            and measurement.get('active_leases')==0,
        'quality':quality.get('passed') if quality.get('ground_truth_sha256') else None,
        'five_repetitions':profile.repetitions>=5,
        'service_limits':None,'matched_reference':None,'measured_advantage':None,
        'fairness':measurement.get('bounded_progress_passed'),
        'dynamic_membership':measurement.get('dynamic_membership_passed')
            if profile.slots>1 else True,
        'task_lifecycle':measurement.get('task_lifecycle_passed'),
        'task_quality_coverage':quality.get('coverage_passed'),
        'actual_batched_forward':measurement.get('actual_max_forward_batch',0)>1
            if profile.slots>1 else True}
    if 'response_p95_ms' in limits and 'response_p95_ms' in measurement:
        checks['service_limits']=measurement['response_p95_ms']<=limits['response_p95_ms']
    if reference is not None:
        paired=reference.get('profile',{})
        keys=('model_type','task','model_id','revision','inputs_sha256','execution_dtype',
              'repetitions','warmups','settings','validation_kind')
        checks['matched_reference']=all(paired.get(k)==getattr(profile,k) for k in keys)
        checks['matched_reference'] &= all(reference.get('source',{}).get(k)==
            measurement.get('source',{}).get(k)
            for k in ('git_revision','source_sha256','packages'))
        checks['matched_reference'] &= reference.get('hardware')==measurement.get('hardware')
        baseline=reference.get('throughput_per_second')
        target=measurement.get('throughput_per_second')
        required=limits.get('minimum_throughput_ratio')
        if baseline and target is not None and required is not None:
            checks['measured_advantage']=target/baseline>=required
    failed=[k for k,v in checks.items() if v is False]
    missing=[k for k,v in checks.items() if v is None]
    return {'schema_version':TASK_GATE_VERSION,'status':'failed' if failed else
            'pending' if missing else 'passed','checks':checks,'failed':failed,'pending':missing,
            'qualification_promoted':False}


def validate_task_evidence(result,scope,evidence,*,performance=False):
    """Existing feature registry calls this validator for task-scoped claims."""
    from mlx_one.registry import RegistryError

    required={'manifest.json','measurement.json','checkpoint.json','task-parity.json',
              'task-quality.json','task-gates.json','preflight.json','supervision.json'}
    documents={Path(a.uri).name:json.loads(Path(a.uri).read_text()) for a in evidence[1:]
               if Path(a.uri).name in required}
    if not required<=documents.keys():
        raise RegistryError('task qualification requires complete checksummed artifacts')
    manifest=documents['manifest.json']
    measurement=documents['measurement.json']
    checkpoint=documents['checkpoint.json']
    try:
        profile=TaskProfile(**manifest['profile'])
    except (KeyError,TypeError,ValueError) as error:
        raise RegistryError('task qualification profile is invalid') from error
    if result.get('feature')!='task:'+profile.task:
        raise RegistryError('task feature does not match its profile')
    current=source_provenance(Path.cwd())
    preflight_report=documents['preflight.json']
    supervision=documents['supervision.json']
    if (preflight_report.get('status')!='PASS' or supervision.get('status')!='completed'
            or supervision.get('exit_code')!=0 or not supervision.get('rss_samples')
            or measurement.get('hardware')!=preflight_report.get('hardware')
            or scope.hardware!=preflight_report.get('hardware',{}).get('chip')
            or not preflight_report.get('gpu_smoke',{}).get('metal')):
        raise RegistryError(
            'native task preflight/supervision does not establish hardware execution')
    if (result.get('task_evidence_version')!=TASK_EVIDENCE_VERSION
            or result.get('validation_kind')!='native-hardware'
            or manifest.get('profile_sha256')!=profile.digest
            or any(manifest.get('source',{}).get(k)!=current.get(k) for k in
                ('git_revision','source_sha256','packages'))
            or scope.source_revision!=current.get('git_revision')
            or scope.model_id!=profile.model_id or scope.revision!=profile.revision
            or profile.validation_kind!='checkpoint' or checkpoint.get('revision')!=profile.revision
            or not checkpoint.get('files') or any(not re.fullmatch('[0-9a-f]{64}',v)
                for v in checkpoint['files'].values())
            or not any(n.endswith('.safetensors') for n in checkpoint['files'])):
        raise RegistryError(
            'task evidence is stale, synthetic or has incomplete checkpoint identity')
    gates=task_gates(profile,documents['task-parity.json'],measurement,
                     documents['task-quality.json'],reference=measurement.get('reference'))
    required_checks=TASK_CORRECTNESS_CHECKS
    if performance:
        required_checks=set(gates['checks'])
    if (documents['task-gates.json'].get('schema_version')!=TASK_GATE_VERSION
            or not all(gates['checks'].get(k) is True for k in required_checks)):
        raise RegistryError('required task correctness/quality/service gates have not passed')


def export_task_matrix(directories,output):
    rows=[]
    for directory in directories:
        root=Path(directory)
        validate_integrity(root,required=('manifest.json',))
        manifest=json.loads((root/'manifest.json').read_text())
        failure=(json.loads((root/'failure.json').read_text())
                 if (root/'failure.json').exists() else {})
        gates=json.loads((root/'task-gates.json').read_text()) if (
            root/'task-gates.json').exists() else {}
        profile=manifest.get('profile',{})
        if failure.get('status') and failure['status'] not in STATUSES:
            raise ValueError('unknown qualification matrix status')
        rows.append({'model_type':profile.get('model_type'),'task':profile.get('task'),
            'status':failure.get('status') or ('FAIL' if gates.get('status')=='failed' else
                'PASS' if gates.get('status')=='passed' else 'NOT_RUN'),
            'reason':failure.get('reason') or ', '.join(
                gates.get('failed',[])+gates.get('pending',[])),
            'source':manifest.get('source',{}).get('git_revision'),'artifact':str(root),
            'qualification_promoted':False})
    accounted={(r['model_type'],r['task']) for r in rows}
    for entry in runtime_coverage():
        for task in entry['tasks'] or (None,):
            if (entry['model_type'],task) not in accounted:
                rows.append({'model_type':entry['model_type'],'task':task,
                    'status':'SKIPPED' if entry['component_only'] else 'NOT_RUN'
                        if entry['shared_service_integrated'] else 'BLOCKED',
                    'reason':entry['reason'],'source':None,'artifact':None,
                    'qualification_promoted':False})
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    write(output/'task-matrix.json',{'rows':rows,'qualification_promoted':False})
    with (output/'task-matrix.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output/'task-matrix.md').write_text('| Model | Task | Status | Reason |\n|---|---|---|---|\n'+
        ''.join(f"| {r['model_type']} | {r['task']} | {r['status']} | {r['reason']} |\n"
                for r in rows))
    return rows


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('inventory')
    run=commands.add_parser('run')
    run.add_argument('--profile',required=True)
    run.add_argument('--output')
    plan=commands.add_parser('plan')
    plan.add_argument('--profile',required=True)
    matrix=commands.add_parser('matrix')
    matrix.add_argument('directories',nargs='*')
    matrix.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    if args.command=='inventory':
        print(json.dumps(task_inventory(),indent=2))
    elif args.command=='plan':
        item=read_profile(args.profile)
        print(json.dumps({'profile':asdict(item),'profile_sha256':item.digest,
            'command':[sys.executable,'benchmarks/native_tasks.py','run','--profile',args.profile],
            'status':'NOT_RUN','qualification_promoted':False},indent=2))
    elif args.command=='run':
        item=read_profile(args.profile)
        root=(Path(args.output) if args.output else Path('qualification/results')/
              socket.gethostname()/item.model_type/str(time.time_ns()))
        print(run_task_profile(item,root))
    else:
        export_task_matrix(args.directories,args.output)
    return 0
