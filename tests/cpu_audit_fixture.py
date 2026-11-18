"""Test substitutes only for copied CPU fixture modules and local CONFIG data."""
import hashlib
import json
from pathlib import Path

OPERATION_MODULES=tuple("scripts/"+name+".py" for name in (
    "prepare_dh116_pinch_margin_contact", "prepare_dh116_pinch_segment_contact",
    "prepare_dh116_right_pinch_native_interval", "prepare_dh116_bilateral_owned_native_interval",
    "prepare_dh116_effective_right_native", "prepare_dh116_tracking_pinch_native",
    "prepare_dh116_left_pinch_approach_native", "prepare_dh116_projected_pinch_native",
    "prepare_dh116_health_projected_pinch", "prepare_dh116_mesh_reset_health_projected_pinch"))
PARENT_AUDIT="scripts/validate_dh116_native_parameter_task.py"
HEALTH_AUDIT="scripts/validate_dh116_health_projected_pinch.py"


def require_temporary_project(project):
    project=Path(project).resolve();git_marker=project/".git"
    if (project==Path(__file__).resolve().parents[1]
            or git_marker.exists() or git_marker.is_symlink()):
        raise ValueError("temporary CPU fixture project required; Git worktrees forbidden")
    return project


def copied_path(project,relative):
    path=(project/relative).resolve()
    if not path.is_relative_to(project):
        raise ValueError("CPU fixture path escapes project")
    return path


def install_local_pair_loaders(project,modules=OPERATION_MODULES):
    project=require_temporary_project(project)
    identities={str(path.relative_to(project)):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (project/"configs").rglob("*.json")}
    template=r"""
# CPU fixture only: preserve scenario_scope bindings, replace archived derivation.
_CPU_FIXTURE_PROJECT=PROJECT_VALUE
_CPU_FIXTURE_CONFIG_SHA=CONFIG_VALUE
def load_pair(project=ROOT):
    import hashlib,json
    from pathlib import Path
    project=Path(project).resolve()
    if project!=Path(_CPU_FIXTURE_PROJECT):
        raise ValueError('CPU fixture project identity mismatch')
    config=(project/CONFIG).resolve()
    if not config.is_relative_to(project):
        raise ValueError('CPU fixture CONFIG escapes project')
    relative=str(config.relative_to(project));payload=config.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=_CPU_FIXTURE_CONFIG_SHA.get(relative):
        raise ValueError('CPU fixture local CONFIG identity changed')
    with (project/'cpu_fixture_pair_reads.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(cpu_fixture=True,config=relative,project=str(project)))+'\n')
    return json.loads(payload)
""".replace("PROJECT_VALUE",repr(str(project))).replace("CONFIG_VALUE",repr(identities))
    for relative in modules:
        path=copied_path(project,relative)
        path.write_text(path.read_text()+template)


def instrument_cpu_audit(project,relative):
    project=require_temporary_project(project);path=copied_path(project,relative)
    template=r"""
_CPU_FIXTURE_AUDIT_ORIGINAL=audit_outcome
def audit_outcome(*args,**kwargs):
    import json
    from pathlib import Path
    project=Path(PROJECT_VALUE);output=Path(args[0])
    before=not any((project/'private'/'native-attempts').glob('**/receipt.json'))
    assert before, 'CPU fixture audit must run before receipt'
    result=_CPU_FIXTURE_AUDIT_ORIGINAL(*args,**kwargs)
    assert result.get('cpu_fixture') is True, 'CPU fixture audit result required'
    reject=(project/'reject').exists()
    assert result.get('lifecycle_qualified') is (not reject), 'CPU fixture intended audit result required'
    reason='cpu_fixture_requested_rejection' if reject else 'cpu_fixture_audit_accepted'
    event=dict(cpu_fixture=True,module=MODULE_VALUE,before_receipt=before,reject_requested=reject,reason=reason)
    with (output/'cpu_fixture_audit_calls.jsonl').open('a') as stream:
        stream.write(json.dumps(event)+'\n')
    return dict(result,cpu_fixture_audit_invoked=True,reason=reason)
""".replace("PROJECT_VALUE",repr(str(project))).replace("MODULE_VALUE",repr(relative))
    path.write_text(path.read_text()+template)


def install_cpu_audit_dependencies(project,*,health=False):
    install_local_pair_loaders(project)
    modules=[PARENT_AUDIT]+([HEALTH_AUDIT] if health else [])
    for relative in modules:
        instrument_cpu_audit(project,relative)


def assert_cpu_audit_artifacts(project,child,reject,*,health=False):
    """Synthetic audit transport checks confer no native qualification."""
    project=Path(project)
    assert child['native_execution_performed'] is False
    assert child['qualification_scope']=='cpu_fixture_only'
    output=Path(child['output'])
    events=[json.loads(line) for line in (output/'cpu_fixture_audit_calls.jsonl').read_text().splitlines()]
    expected=[PARENT_AUDIT]+([HEALTH_AUDIT] if health else [])
    assert [event['module'] for event in events]==expected
    reason='cpu_fixture_requested_rejection' if reject else 'cpu_fixture_audit_accepted'
    assert all(event['cpu_fixture'] is True and event['before_receipt'] is True
        and event['reject_requested'] is reject and event['reason']==reason for event in events)
    receipt=next(Path(child['worker_private']).glob('native-attempts/*/*/*/receipt.json'))
    artifacts=[json.loads(path.read_text()) for path in receipt.parent.glob('*audit.json')]
    audits=[a['evidence'] for a in artifacts if a.get('evidence',{}).get('cpu_fixture_audit_invoked') is True]
    assert len(audits)==1
    evidence=audits[0]
    assert evidence['cpu_fixture'] is True and evidence['lifecycle_qualified'] is (not reject)
    assert evidence['reason']==reason
    if reject:
        assert child['reason']=='lifetime outcome is diagnostic only: '+reason
