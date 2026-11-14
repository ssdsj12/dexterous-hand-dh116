"""CPU-only closure worker gates; no native application or solver is constructed."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_dh116_pinch_v3 import cpu
from scripts import verify_dh116_dual_pinch_closure_interval as entry
from scripts import run_dh116_dual_pinch_closure_work_packages as wrapper

ROOT = Path(__file__).resolve().parents[1]
SELF = "tests/test_dh116_dual_pinch_closure_worker.py"
REQUIRED = {
    "scripts/dh116_dual_pinch_lifetime_evidence.py",
    "src/go2_pvcnn/control/dh116/tabletop_measured_closure_geometry.py",
    "tests/test_dh116_measured_closure_geometry.py",
    "scripts/prepare_dh116_dual_pinch_readback_profile.py",
    "scripts/prepare_dh116_pinch_closure_margin.py",
    "scripts/solve_dh116_tripod_sdf_constrained_direction.py",
    "scripts/audit_dh116_power_closure_constraints.py",
    "scripts/screen_dh116_right_pinch_closure.py",
    "tests/test_dh116_dual_pinch_readback_profile.py", SELF,
}
BLOCKER = """
import sys, importlib.abc
forbidden = {'pxr','PhysxSchema','isaaclab','isaacsim','omni','carb','ropesim'}
class RefuseNativeImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in forbidden:
            raise RuntimeError('CPU_FORBIDDEN_IMPORT:' + fullname)
sys.meta_path.insert(0, RefuseNativeImport())
"""


def test_real_preflight_in_clean_process_blocks_all_native_imports():
    done = cpu(BLOCKER + """
from pathlib import Path
from scripts import verify_dh116_dual_pinch_closure_interval as entry
from scripts import run_dh116_dual_pinch_closure_work_packages as wrapper
value = entry.preflight(Path.cwd())
entry.original.prior.assert_clean_import_boundary()
assert not forbidden.intersection({name.split('.')[0] for name in sys.modules})
assert value['profile_sha256'] == entry.PROFILE_SHA
assert value['physics_steps'] == 18 and value['native_execution_performed'] is False
print('CLOSURE_BYTE_ONLY')
""")
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.strip() == "CLOSURE_BYTE_ONLY"


@pytest.fixture(scope="module")
def contract():
    return entry.preflight(ROOT)


def test_actual_parent_profile_and_all_source_pins(contract):
    profile = entry.load_profile()
    assert contract['profile_sha256'] == entry.PROFILE_SHA == "67fe4ac5a231355aac627d2ce4e7bd8186b138273d80df3c30a5c5406c4e0498"
    assert contract['native_parent_receipt_sha256'] == entry.PARENT_SHA
    assert profile['native_parent_receipt_path'] == str(entry.PARENT_RECEIPT)
    assert REQUIRED <= set(entry.FILES) <= set(contract['sources']['local_sources'])
    for name in REQUIRED:
        assert contract['sources']['local_sources'][name] == entry.sha(ROOT/name)
    expected = {
        str(entry.PROFILE): entry.PROFILE_SHA,
        str(entry.PROFILE.with_name('result.json')): entry.RESULT_SHA,
        str(entry.PARENT_RECEIPT): entry.PARENT_SHA,
        str(entry.PARENT_RECEIPT.with_name('dual-pinch-interval-post-exit-audit.json')): entry.PARENT_AUDIT_SHA,
        profile['native_parent_ledger_path']: entry.LEDGER_SHA,
    }
    assert expected.items() <= contract['sources']['external_sources'].items()


@pytest.mark.parametrize('wrong', ['path','sha'])
def test_other_parent_identity_is_refused(wrong):
    with pytest.raises(ValueError, match='exact published'):
        entry.load_profile(entry.PARENT_RECEIPT.with_name('other.json') if wrong=='path' else entry.PARENT_RECEIPT,
                           '0'*64 if wrong=='sha' else entry.PARENT_SHA)


@pytest.mark.parametrize('evidence', ['parent','audit','profile','result','ledger'])
def test_real_evidence_byte_identity_is_checked_without_mutating_files(monkeypatch,evidence):
    ledger = Path(entry.read(entry.PARENT_RECEIPT)['output'])/'physical.jsonl'
    target = dict(parent=entry.PARENT_RECEIPT,
        audit=entry.PARENT_RECEIPT.with_name('dual-pinch-interval-post-exit-audit.json'),
        profile=entry.PROFILE,result=entry.PROFILE.with_name('result.json'),ledger=ledger)[evidence]
    real = entry.sha
    monkeypatch.setattr(entry,'sha',lambda path: '0'*64 if Path(path)==target else real(path))
    with pytest.raises(ValueError,match='differ'):
        entry.load_profile()


@pytest.mark.parametrize('name', sorted(REQUIRED))
def test_new_source_tamper_changes_code_identity(monkeypatch,contract,name):
    real = entry.sha
    old = deepcopy(contract)
    monkeypatch.setattr(entry,'sha',lambda path: '0'*64 if Path(path)==ROOT/name else real(path))
    changed = entry.preflight(ROOT)
    assert changed['sources']['local_sources'][name] == '0'*64
    assert changed['code_config_sha256'] != contract['code_config_sha256']
    assert old == contract


def marker(output):
    return dict(version=entry.VERSION,case='explicit',output=str(output.resolve()),
        entry_sha256=entry.sha(ROOT/entry.ENTRY),wrapper_sha256=entry.sha(ROOT/entry.WRAPPER))


def worker_bindings():
    return entry.bindings((entry.original,'ENTRY',entry.ENTRY),(entry.original,'WRAPPER',entry.WRAPPER),
        (entry.original,'VERSION',entry.VERSION),(entry.original,'WORKER_ENV',entry.WORKER_ENV))


@pytest.mark.parametrize('damage',[None,'missing','output','version','entry_sha256','wrapper_sha256','gpu','gdb'])
def test_dedicated_child_marker_and_gpu3_stopped_gdb(monkeypatch,tmp_path,damage):
    output = tmp_path/'not-created'
    value = marker(output)
    if damage in ('output','version','entry_sha256','wrapper_sha256'):value[damage]='wrong'
    monkeypatch.setenv(entry.WORKER_ENV,json.dumps(value))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','2' if damage=='gpu' else '3')
    monkeypatch.setenv('DH116_PHYSX_LIFETIME_GDB','0' if damage=='gdb' else '1')
    if damage=='missing':monkeypatch.delenv(entry.WORKER_ENV)
    with worker_bindings():
        if damage is None:entry.original.require_worker(SimpleNamespace(output=output),ROOT)
        else:
            with pytest.raises(ValueError,match='worker|GPU|debugger'):
                entry.original.require_worker(SimpleNamespace(output=output),ROOT)
    assert not output.exists()


def recipe():
    spec = deepcopy(entry.read(ROOT/'configs/dh116_execution/deformx_dual_pinch_zero_interval_v4.json'))
    spec['entrypoint'] = entry.ENTRY
    spec['source_files'] = sorted(set(spec['source_files']) | set(wrapper.AUDIT_SOURCES))
    spec['budget'].update(physics_steps=18,updates=0,environments=1,gpu_index=3,timeout_seconds=180)
    spec['argv'][spec['argv'].index('--steps')+1] = '18'
    spec['argv'][spec['argv'].index('--approved-code-config-sha256')+1] = 'a'*64
    spec['validator']['equals'].update(version=entry.VERSION,code_config_sha256='a'*64,
        physics_steps=18,measurement_steps=16,rod_steps=16)
    spec['dependency_receipt'] = dict(package=entry.PARENT_PACKAGE,qualification_scope=entry.PARENT_SCOPE)
    return spec


def cleanup(spec):
    with entry.bindings((wrapper.base,'ENTRY',entry.ENTRY),(wrapper.base,'AUDIT_SOURCES',wrapper.AUDIT_SOURCES)):
        return wrapper.cleanup_case(spec)


def test_cpu_recipe_uses_real_base_cleanup_without_future_recipe_hash():
    assert cleanup(recipe()) == 'explicit'


@pytest.mark.parametrize('field,value',[('physics_steps',17),('physics_steps',19),('updates',1),
    ('environments',2),('gpu_index',2),('timeout_seconds',179),('timeout_seconds',181)])
def test_cleanup_budget_cannot_expand_or_shrink(field,value):
    spec = recipe();spec['budget'][field] = value
    with pytest.raises(ValueError):cleanup(spec)


@pytest.mark.parametrize('damage',['dependency','scope','package','entry','source','debugger',
    'case','code','code_missing','code_duplicate','version'])
def test_cleanup_preserves_source_and_parent_scope(damage):
    spec = recipe()
    if damage=='dependency':spec.pop('dependency_receipt')
    elif damage in ('scope','package'):
        spec['dependency_receipt']['qualification_scope' if damage=='scope' else 'package']='wrong'
    elif damage=='entry':spec['entrypoint']='scripts/verify_dh116_dual_pinch_interval_v4.py'
    elif damage=='source':spec['source_files'].remove('scripts/dh116_dual_pinch_lifetime_evidence.py')
    elif damage=='debugger':spec['debugger']='none'
    elif damage=='case':spec['argv'][spec['argv'].index('--cleanup')+1]='registry'
    elif damage=='code':spec['validator']['equals']['code_config_sha256']='b'*64
    elif damage=='version':spec['validator']['equals']['version']='previous'
    elif damage=='code_missing':
        index=spec['argv'].index('--approved-code-config-sha256');del spec['argv'][index:index+2]
    elif damage=='code_duplicate':spec['argv'] += ['--approved-code-config-sha256','a'*64]
    with pytest.raises(ValueError):cleanup(spec)


@pytest.mark.parametrize('fail',[False,True])
def test_wrapper_reentry_and_bindings_restore_on_success_or_exception(monkeypatch,fail):
    names=('__file__','ENTRY','WRAPPER','VERSION','WORKER_ENV','AUDIT_NAME','AUDIT_SOURCES',
           'cleanup_case','render_gdb_commands','audit_outcome')
    original={name:getattr(wrapper.base,name) for name in names}
    def child():
        assert wrapper.base.__file__ == str(Path(wrapper.__file__).resolve())
        assert wrapper.base.ENTRY == entry.ENTRY and wrapper.base.WRAPPER == entry.WRAPPER
        assert wrapper.base.WORKER_ENV == entry.WORKER_ENV and wrapper.base.VERSION == entry.VERSION
        assert wrapper.base.cleanup_case is wrapper.cleanup_case
        assert wrapper.base.render_gdb_commands is wrapper.render_gdb_commands
        assert wrapper.base.audit_outcome is wrapper.audit_outcome
        assert set(entry.FILES) <= set(wrapper.base.AUDIT_SOURCES)
        if fail:raise RuntimeError('CPU child sentinel')
        return 37
    monkeypatch.setattr(wrapper.base,'main',child)
    if fail:
        with pytest.raises(RuntimeError,match='CPU child sentinel'):wrapper.main()
    else:assert wrapper.main()==37
    assert {name:getattr(wrapper.base,name) for name in names}==original


@pytest.mark.parametrize('damage',[None,'code','steps'])
def test_real_cli_child_stops_at_cpu_execute_stub_before_any_native_resource(tmp_path,damage):
    code = BLOCKER + """
import os,json
from pathlib import Path
from scripts import verify_dh116_dual_pinch_closure_interval as entry
output = Path(OUTPUT)
contract = entry.preflight(Path.cwd())
marker = dict(version=entry.VERSION,case='explicit',output=str(output.resolve()),
    entry_sha256=entry.sha(Path.cwd()/entry.ENTRY),wrapper_sha256=entry.sha(Path.cwd()/entry.WRAPPER))
os.environ[entry.WORKER_ENV] = json.dumps(marker)
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['DH116_PHYSX_LIFETIME_GDB'] = '1'
observed = []
def cpu_execute(args,project,actual):
    assert actual == contract and args.steps == 18
    assert not forbidden.intersection({name.split('.')[0] for name in sys.modules})
    observed.append(True)
entry.execute = cpu_execute
sys.argv = [entry.ENTRY,'--cleanup','explicit','--output',str(output),'--headless',
    '--execute-native','--steps','17' if DAMAGE=='steps' else '18',
    '--approved-code-config-sha256','0'*64 if DAMAGE=='code' else contract['code_config_sha256'],
    '--left-scheduler-receipt',str(entry.PARENT_RECEIPT),'--left-receipt-sha256',entry.PARENT_SHA]
try: entry.main()
except SystemExit as error:
    assert DAMAGE in ('code','steps') and error.code==2
    assert not output.exists() and not observed
else:
    assert DAMAGE is None and observed==[True] and output.is_dir()
print('CPU_CLI_GATE_CHECKED')
"""
    code=code.replace('OUTPUT',repr(str(tmp_path/'cpu-child'))).replace('DAMAGE',repr(damage))
    done=cpu(code)
    assert done.returncode==0,done.stdout+done.stderr
    assert done.stdout.strip()=='CPU_CLI_GATE_CHECKED'



def test_actual_frozen_recipe_matches_current_preflight_sources_and_parent():
    # Read the finished recipe dynamically: this test itself participates in its source SHA.
    spec = entry.read(ROOT/'configs/dh116_execution/deformx_dual_pinch_guarded_closure_v1.json')
    current = entry.preflight(ROOT)
    args = spec['argv']
    assert args.count('--approved-code-config-sha256') == 1
    code = args[args.index('--approved-code-config-sha256')+1]
    assert code == spec['validator']['equals']['code_config_sha256'] == current['code_config_sha256']
    assert spec['entrypoint'] == entry.ENTRY and cleanup(spec) == 'explicit'
    assert args.count('--steps') == 1 and args[args.index('--steps')+1] == '18'
    assert args[args.index('--left-scheduler-receipt')+1] == '{left_scheduler_receipt}'
    assert args[args.index('--left-receipt-sha256')+1] == '{left_receipt_sha256}'
    assert spec['dependency_receipt'] == dict(package=entry.PARENT_PACKAGE,qualification_scope=entry.PARENT_SCOPE)
    assert spec['validator']['line_counts']['sdk_clock.jsonl'] == 16
    assert spec['validator']['physics_clock'] == dict(ledger='sdk_clock.jsonl',setup_steps=2)
    expected = dict(version=entry.VERSION,physics_steps=18,measurement_steps=16,rod_steps=16,ppo_updates=0)
    assert expected.items() <= spec['validator']['equals'].items()
    assert entry.WORKER_ENV not in spec['environment']
    source_files = set(spec['source_files'])
    assert len(source_files) == len(spec['source_files'])
    assert {entry.ENTRY,entry.WRAPPER,SELF} | set(entry.FILES) | set(wrapper.AUDIT_SOURCES) <= source_files
    for name in source_files:
        assert not Path(name).is_absolute() and '..' not in Path(name).parts and (ROOT/name).is_file()
    from scripts.run_dh116_work_packages import release_path
    external = current['sources']['external_sources']
    assert external.items() <= spec['inputs'].items()
    extras = {path:digest for path,digest in spec['inputs'].items() if path not in external}
    readonly = {}
    for name,digest in current['sources']['local_sources'].items():
        try:release_path(name)
        except ValueError:
            assert name not in source_files
            matches = [path for path,actual in spec['inputs'].items()
                       if Path(path).is_absolute() and path.endswith('/'+name) and actual==digest]
            assert len(matches)==1, name
            assert entry.sha(Path(matches[0]))==digest
            readonly[matches[0]]=digest
        else:assert name in source_files, name
    assert readonly and extras == {path:digest for path,digest in readonly.items() if path not in external}
