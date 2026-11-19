"""Strict companion contract and pre-App worker entry for actual execution.

Parsing is CPU-only. The companion never grants authority: current registered
intent, complete source inventory and a parent-issued kernel peer lease do.
"""
from copy import deepcopy
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from .tabletop_native_teacher_worker_contract import validate_teacher_recipe_v2

SCHEMA = 'dh116-actual-native-execution-v1'
IDENTITY_SCHEMA = 'dh116-actual-native-execution-identity-v1'
PROVIDER = 'src/go2_pvcnn/control/dh116/tabletop_native_teacher_actual_provider.py'
ENVIRONMENT = 'configs/dh116_execution/actual-environment-v1.json'
SOURCE_FILES = frozenset({
    'src/go2_pvcnn/control/dh116/tabletop_native_teacher_actual_launch.py',
    'src/go2_pvcnn/control/dh116/tabletop_native_teacher_actual_process.py',
    'scripts/train_dh116_native_teacher_worker.py',
    'tests/test_native_teacher_actual_launch.py',
    'tests/test_native_teacher_actual_launch_process.py',
})
_SHA = re.compile(r'[0-9a-f]{64}')

def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def _sha(value):
    return type(value) is str and _SHA.fullmatch(value) is not None

def _regular(root, name):
    path = Path(name)
    if not path.is_absolute():path = root/path
    if path.resolve(strict=True) != path or not path.is_file():
        raise ValueError('actual launch requires regular source/input files')
    return path

def _local_import_closure(root, roots):
    """Audit local Python dependencies, including imports inside functions.

    The sole dynamic provider is a fixed path explicitly supplied as a root.
    External installed libraries remain under the existing backend profile.
    """
    pending = list(roots)
    seen = set()
    checked = set(roots)

    def enqueue_if_local(name):
        if name not in checked:
            checked.add(name)
            if (root / name).is_file():
                pending.append(name)

    while pending:
        name=pending.pop()
        if name in seen or not name.endswith('.py'):continue
        seen.add(name)
        path=_regular(root,name)
        for parent in Path(name).parents:
            init=(parent/'__init__.py').as_posix()
            if init != '__init__.py':
                enqueue_if_local(init)
        module=name.removesuffix('.py').replace('/','.')
        if module.startswith('src.'):module=module[4:]
        package=module.rsplit('.',1)[0]
        for node in ast.walk(ast.parse(path.read_bytes(),filename=str(path))):
            candidates=[]
            if isinstance(node,ast.Import):candidates=[item.name for item in node.names]
            elif isinstance(node,ast.ImportFrom):
                prefix=package.split('.')
                if node.level:
                    prefix=prefix[:len(prefix)-node.level+1]
                    base='.'.join([*prefix,*([node.module] if node.module else [])])
                else:base=node.module or ''
                candidates=[base,*[base+'.'+item.name for item in node.names if item.name!='*']]
            for candidate in candidates:
                for base in ('src/',''):
                    relative = base + candidate.replace('.', '/')
                    enqueue_if_local(relative + '.py')
                    enqueue_if_local(relative + '/__init__.py')
    return seen

def audit_loaded_actual_sources(repository, package):
    """Reject undeclared local production modules, including dynamic imports."""
    from scripts.run_dh116_work_packages import native_recipe_sources
    inventory=native_recipe_sources(repository,package,repository.native_recipe(package)['spec'])
    for module in tuple(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if not filename:continue
        path=Path(filename).resolve()
        if path.is_relative_to(repository.root):
            relative=path.relative_to(repository.root).as_posix()
            if relative.startswith(('src/','scripts/')) and relative.endswith('.py') and relative not in inventory:
                raise ValueError('actual loaded producer source closure incomplete: '+relative)

def validate_actual_launch(launch, *, repository, package, recipe=None):
    """Validate current companion bytes against the unchanged C06 core."""
    recipe=recipe or repository.native_recipe(package)
    spec=validate_teacher_recipe_v2(recipe['spec'])
    required={'schema','core_recipe_identity','source_closure_sha256','role','device_plan','debugger','provider','checkpoint_origin'}
    core={k:recipe[k] for k in ('command_spec_sha256','code_config_sha256')}
    if (type(launch) is not dict or set(launch)!=required or launch['schema']!=SCHEMA
            or launch['core_recipe_identity']!=core or launch['debugger']!='gdb_on_segv'
            or launch['source_closure_sha256']!=spec['teacher_contract']['source_sha256']):
        raise ValueError('actual launch core/source/debugger binding invalid')
    if launch['role']=='b_resume':raise ValueError('actual_resume_not_implemented')
    if launch['role']!='a_fresh' or launch['checkpoint_origin'] is not None:
        raise ValueError('actual launch role/origin invalid')
    provider=launch['provider']
    if (type(provider) is not dict or set(provider)!={'path','entrypoint','sha256'}
            or provider['path']!=PROVIDER or provider['entrypoint']!='run_a' or not _sha(provider['sha256'])):
        raise ValueError('actual launch fixed provider binding invalid')
    plan=launch['device_plan']
    if (type(plan) is not dict or set(plan)!={'physical_gpu_index','cuda_visible_devices','logical_device','experience_path','experience_sha256'}
            or type(plan['physical_gpu_index']) is not int or plan['physical_gpu_index']!=3
            or spec['budget']['gpu_index']!=plan['physical_gpu_index']
            or plan['cuda_visible_devices']!='3' or plan['logical_device']!='cuda:0'
            or type(plan['experience_path']) is not str or not Path(plan['experience_path']).is_absolute()
            or not _sha(plan['experience_sha256'])):
        raise ValueError('actual launch explicit physical3/mask3/logical0 device plan invalid')
    from scripts.run_dh116_work_packages import file_hash,native_recipe_sources,GDB
    if ENVIRONMENT not in spec['source_files']:
        raise ValueError('actual core environment source binding missing')
    environment=json.loads(_regular(repository.root,ENVIRONMENT).read_bytes())
    expected_experience=dict(path=plan['experience_path'],sha256=plan['experience_sha256'])
    if (type(environment) is not dict or set(environment)!={'schema','experience','debugger'}
            or environment['schema']!='dh116-actual-native-environment-v1'
            or environment['experience']!=expected_experience
            or environment['debugger']!={'path':GDB,'sha256':file_hash(Path(GDB))}):
        raise ValueError('actual core environment experience/debugger source differs')
    if file_hash(_regular(repository.root,plan['experience_path']))!=plan['experience_sha256']:
        raise ValueError('actual launch experience source differs')
    if not (repository.root/PROVIDER).exists():raise ValueError('actual_provider_not_implemented')
    if file_hash(_regular(repository.root,PROVIDER))!=provider['sha256']:
        raise ValueError('actual launch provider source differs')
    inventory=native_recipe_sources(repository,package,spec)
    needed=SOURCE_FILES|{PROVIDER}|_local_import_closure(repository.root,SOURCE_FILES|{PROVIDER})
    if not needed<=inventory.keys():
        raise ValueError('actual launch producer source closure incomplete: '+', '.join(sorted(needed-inventory.keys())))
    if hashlib.sha256(json.dumps(inventory,sort_keys=True).encode()).hexdigest()!=launch['source_closure_sha256']:
        raise ValueError('actual launch source inventory differs')
    # Companion execution metadata must stay outside the core source inventory.
    binding=package.get('actual_launch')
    if binding and binding.get('path') in inventory:raise ValueError('actual companion self-hash cycle forbidden')
    return deepcopy(launch)

def actual_launch_identity(launch, *, companion_sha256):
    if not _sha(companion_sha256):raise ValueError('actual companion SHA required')
    return dict(schema=IDENTITY_SCHEMA,core_recipe_identity=deepcopy(launch['core_recipe_identity']),
        companion_sha256=companion_sha256,source_closure_sha256=launch['source_closure_sha256'])

def _pci(value):
    if type(value) is not str or not re.fullmatch(r'[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]',value):
        raise ValueError('actual_device_identity_unverified')
    return tuple(int(part,16) for part in re.split(r'[:.]',value))

def validate_actual_device(plan, *, cuda_visible_devices, physical_gpu_indices, logical_device_count,
        physical_pci_bus_id=None, logical_pci_bus_id=None):
    if (type(physical_gpu_indices) is not list or any(type(x) is not int for x in physical_gpu_indices)
            or type(logical_device_count) is not int or logical_device_count!=1
            or cuda_visible_devices!=plan['cuda_visible_devices'] or cuda_visible_devices!='3'
            or plan['physical_gpu_index'] not in physical_gpu_indices or plan['logical_device']!='cuda:0'
            or _pci(physical_pci_bus_id)!=_pci(logical_pci_bus_id)):
        raise ValueError('actual_device_identity_unverified')
    return deepcopy(plan)

def _probe_actual_device(plan):
    # Installed nvidia-smi supplies physical indices; CUDA's installed driver
    # supplies the actual visible count after the approved mask is applied.
    import ctypes
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,pci.bus_id','--format=csv,noheader,nounits'],text=True,timeout=5)
    physical={int(fields[0].strip()):fields[1].strip() for line in raw.splitlines() if line.strip() for fields in [line.split(',')]}
    driver=ctypes.CDLL('libcuda.so.1');count=ctypes.c_int()
    if driver.cuInit(0)!=0 or driver.cuDeviceGetCount(ctypes.byref(count))!=0:
        raise ValueError('actual_device_identity_unverified')
    device=ctypes.c_int();bus=ctypes.create_string_buffer(32)
    if driver.cuDeviceGet(ctypes.byref(device),0)!=0 or driver.cuDeviceGetPCIBusId(bus,len(bus),device)!=0:
        raise ValueError('actual_device_identity_unverified')
    return validate_actual_device(plan,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        physical_gpu_indices=list(physical),logical_device_count=count.value,
        physical_pci_bus_id=physical.get(plan['physical_gpu_index']),logical_pci_bus_id=bus.value.decode('ascii'))

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--config-sha256',required=True)
    parser.add_argument('--additional-updates',type=int,required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    try:
        from scripts.run_dh116_work_packages import Repository
        from .tabletop_native_teacher_actual_process import read_launch_binding,request_actual_launch
        from .tabletop_stereo_native import native_instance_lock
        root=Path(__file__).resolve().parents[4]
        binding=dict(path=os.environ.get('DH116_ACTUAL_LAUNCH_PATH',''),sha256=os.environ.get('DH116_ACTUAL_LAUNCH_SHA256',''))
        if not binding['path'] or not _sha(binding['sha256']):raise ValueError('actual_parent_binding_invalid')
        # Infer only the location, not authority, from the canonical attempt
        # path. read_launch_binding then verifies the exact registered intent,
        # current source and live scheduler argv/locks against this Repository.
        path=Path(binding['path'])
        if (not path.is_absolute() or path.resolve(strict=True)!=path or len(path.parents)<6
                or path.name!='binding.json' or path.parent.name!='actual-launch'
                or not path.parents[1].name.startswith('attempt-')
                or not _sha(path.parents[2].name) or path.parents[4].name!='native-attempts'):
            raise ValueError('actual_parent_binding_invalid')
        repository=Repository(root,evidence_dir=path.parents[5])
        # Package selection is from the source-bound envelope, never an import
        # string or worker-provided approval.
        envelope=read_launch_binding(binding,repository=repository)
        package=repository.registry()['packages'][envelope.package_name]
        spec=repository.native_recipe(package)['spec']
        if (args.additional_updates!=1 or args.config!=spec['teacher_contract']['training_config']['path']
                or args.config_sha256!=spec['teacher_contract']['training_config']['sha256']):
            raise ValueError('actual_worker_argv_invalid')
        from scripts.run_dh116_work_packages import NATIVE_LOCK
        with native_instance_lock(NATIVE_LOCK):
            lease=request_actual_launch(binding,repository=repository,package=package)
            try:
                lease.revalidate()
                launch=repository.actual_launch(package)
                audit_loaded_actual_sources(repository,package)
                _probe_actual_device(launch['device_plan'])
                # Only the fixed reviewed module is permitted. Its file SHA
                # and transitive imports were verified before requesting lease.
                import importlib.util
                name=__package__+'.tabletop_native_teacher_actual_provider'
                module_spec=importlib.util.find_spec(name)
                expected=_regular(repository.root,PROVIDER)
                if module_spec is None or Path(module_spec.origin).resolve()!=expected:
                    raise ValueError('actual_provider_source_substitution')
                from . import tabletop_native_teacher_actual_provider as provider
                if Path(provider.__file__).resolve()!=expected:
                    raise ValueError('actual_provider_source_substitution')
                audit_loaded_actual_sources(repository,package)
                provider.run_a(config=lease.process_config,lease=lease,output=Path(args.output))
            finally:lease.close()
        return 0
    except (OSError,ValueError,KeyError,TypeError,ImportError,subprocess.SubprocessError) as error:
        print(str(error),file=sys.stderr)
        return 2
