"""Explicit local plugin validation and atomic architecture/runtime publication.

Metadata validation runs in a separate Python process that refuses execution
backend imports. No model weights, model class or inference backend is loaded.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import asdict

from mlx_one.core.registry import _REGISTRY_LOCK, ModelRegistration, registered_model_types
from mlx_one.engine.adapter_registry import RUNTIME_CONTRACT_VERSION, RuntimeRegistration

PLUGIN_CONTRACT_VERSION='native-registry-plugin-v1'


def _payload(architecture,runtime,config):
    entry=asdict(architecture)
    entry['tasks']=sorted(entry['tasks'])
    entry['capabilities']=sorted(entry['capabilities'])
    return {'schema_version':PLUGIN_CONTRACT_VERSION,'architecture':entry,
            'runtime':asdict(runtime),'config':config}


def _reconstruct(payload):
    if payload.get('schema_version')!=PLUGIN_CONTRACT_VERSION:
        raise ValueError('plugin contract version mismatch')
    entry=dict(payload['architecture'])
    entry['tasks']=frozenset(entry['tasks'])
    entry['capabilities']=frozenset(entry['capabilities'])
    runtime=dict(payload['runtime'])
    runtime['task_adapters']=tuple(tuple(pair) for pair in runtime.get('task_adapters',()))
    return ModelRegistration(**entry),RuntimeRegistration(**runtime)


def _resolve(path):
    module,_,name=path.partition(':')
    return getattr(importlib.import_module(module),name)


def _validate(payload):
    architecture,runtime=_reconstruct(payload)
    if (architecture.model_type!=runtime.model_type
            or runtime.contract_version!=RUNTIME_CONTRACT_VERSION):
        raise ValueError('plugin requires matching current architecture/runtime identity')
    config_class=architecture.config_class()
    if not callable(getattr(config_class,'from_dict',None)):
        raise ValueError('plugin config requires from_dict')
    config=config_class.from_dict(payload['config'])
    if getattr(config,'model_type',None)!=architecture.model_type:
        raise ValueError('plugin config model_type mismatch')
    tasks,states=runtime.specifications(architecture)
    if runtime.integration_status=='integrated':
        if not callable(architecture.loader()):
            raise ValueError('integrated plugin requires callable native loader')
        for task in tasks:
            adapter=runtime.adapter(task.task)
            if any(not callable(getattr(adapter,name,None)) for name in
                   ('describe','estimate','pack','execute','unpack')):
                raise ValueError('plugin adapter does not implement the execution protocol')
    # Inspect selected implementation location without importing native model math.
    module,_,name=architecture.model_path.partition(':')
    spec=importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        raise ValueError('plugin model implementation is missing')
    from pathlib import Path

    source=Path(spec.origin)
    if source.suffix!='.py':
        raise ValueError('plugin template requires an inspectable local Python model')
    tree=ast.parse(source.read_text())
    declared={node.name for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef))}
    declared.update(alias.asname or alias.name for node in tree.body
                    if isinstance(node,ast.ImportFrom) for alias in node.names)
    if name not in declared:
        raise ValueError('plugin model class is not statically declared')
    return {'schema_version':PLUGIN_CONTRACT_VERSION,'model_type':architecture.model_type,
        'tasks':[t.task.value for t in tasks],'state_layers':len(states),
        'metadata_validated':True,'model_executed':False,'checkpoint_validated':False,
        'hardware_qualified':False,'performance_qualified':False}


def validate_plugin(architecture,runtime,*,config):
    """Validate selected metadata under a backend import guard, leaving parent lazy."""
    payload=_payload(architecture,runtime,config)
    environment={**os.environ,'PYTHONPATH':os.pathsep.join(str(p) for p in sys.path if p),
                 'HF_HUB_OFFLINE':'1'}
    process=subprocess.run([sys.executable,'-m','mlx_one.engine.plugins','--validate'],
        input=json.dumps(payload),text=True,capture_output=True,timeout=15,env=environment)
    if process.returncode!=0:
        raise ValueError('isolated plugin metadata validation failed: '+process.stderr.strip())
    report=json.loads(process.stdout)
    if (report.get('schema_version')!=PLUGIN_CONTRACT_VERSION
            or not report.get('metadata_validated')):
        raise ValueError('invalid plugin metadata validation report')
    return report


def register_plugin(architecture,runtime,*,config):
    """Publish both descriptors together; never replace an existing model/runtime."""
    from mlx_one.core import registry
    from mlx_one.engine import adapter_registry

    report=validate_plugin(architecture,runtime,config=config)
    registered_model_types()
    with _REGISTRY_LOCK:
        existing=registry._REGISTRY.get(architecture.model_type)
        bound=adapter_registry._RUNTIME_REGISTRY.get(runtime.model_type)
        if existing is not None or bound is not None:
            if existing==architecture and bound==runtime:
                return report
            raise ValueError('plugin cannot replace an existing architecture/runtime')
        registry._REGISTRY={**registry._REGISTRY,architecture.model_type:architecture}
        adapter_registry._RUNTIME_REGISTRY={**adapter_registry._RUNTIME_REGISTRY,
                                         runtime.model_type:runtime}
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate',action='store_true',required=True)
    parser.parse_args(argv)
    import builtins

    original=builtins.__import__
    def metadata_import(name,*args,**kwargs):
        if name in {'mlx','torch'} or name.startswith(('mlx.','torch.')):
            raise ImportError('plugin metadata cannot import an execution backend')
        return original(name,*args,**kwargs)
    builtins.__import__=metadata_import
    try:
        print(json.dumps(_validate(json.load(sys.stdin))))
    except Exception as error:
        print(str(error),file=sys.stderr)
        return 1
    finally:
        builtins.__import__=original
    return 0


def qualified_task_profile(registry,scope,requested):
    """Opt-in selection only: exact measured task/workload, current source and artifacts."""
    from mlx_one.registry import InferenceFeatureRegistry, InferenceFeatureStatus
    from mlx_one.task_qualification import validate_task_evidence

    fallback={'resolved_profile':None,'reason':'No matching performance-qualified task evidence',
              'qualification_promoted':False}
    if not isinstance(registry,InferenceFeatureRegistry):
        return fallback
    feature='task:'+requested.task
    record=registry.find(feature,scope)
    if record is None or record.status is not InferenceFeatureStatus.PERFORMANCE_QUALIFIED:
        return fallback
    from pathlib import Path

    try:
        if registry.report(feature,scope).get('stale_evidence'):
            return {**fallback,'reason':'Task evidence checksum changed'}
        proof=json.loads(Path(record.evidence[0].uri).read_text())
        validate_task_evidence(proof,scope,record.evidence,performance=True)
        manifest=next(a for a in record.evidence if Path(a.uri).name=='manifest.json')
        measured=json.loads(Path(manifest.uri).read_text())['profile']
        if measured!=asdict(requested):
            return {**fallback,'reason':'Requested task/profile differs from measured workload'}
        return {'resolved_profile':asdict(requested),'reason':'Exact fresh qualified task profile',
                'qualification_promoted':False}
    except (OSError,ValueError,KeyError,StopIteration) as error:
        return {**fallback,'reason':'Task evidence cannot be revalidated: '+str(error)}


if __name__=="__main__":
    raise SystemExit(main())
