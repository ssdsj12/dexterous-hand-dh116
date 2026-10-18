from pathlib import Path
from types import SimpleNamespace
import ast
import pytest

from go2_pvcnn.control.dh116.grasp_sequence import validate_continuous_options


def options(**changes):
    values=dict(grasp_sequence=True,closure_pilot=False,active_pose='pose',active_pose_sha256='sha',
        approach_certificate='cert',approach_certificate_sha256='sha',point_references='refs',
        contact_manifest='mesh',contact_manifest_sha256='sha',disable_contact_reaction=False,
        coordinated_closure=True,gravity_hold=True,table_wire=True,physics_dt=.0001,
        steps=200000,closure_loads=[.87,.77,.88,.93],probe_channel=None)
    values.update(changes);return SimpleNamespace(**values)


def test_valid_continuous_options():
    validate_continuous_options(options())


def test_touch_first_load_floor_checked_before_isaac_launch():
    with pytest.raises(ValueError):
        validate_continuous_options(options(closure_loads=[.049]*4))
    validate_continuous_options(options(closure_loads=[.05]*4))


@pytest.mark.parametrize('change',[
    {'closure_pilot':True},{'active_pose':None},{'active_pose_sha256':None},
    {'approach_certificate':None},{'approach_certificate_sha256':None},
    {'point_references':None},{'contact_manifest':None},{'contact_manifest_sha256':None},
    {'disable_contact_reaction':True},{'coordinated_closure':False},
    {'gravity_hold':False},{'table_wire':False},{'physics_dt':.001},
    {'physics_dt':float('nan')},{'steps':30000},{'closure_loads':None},
    {'closure_loads':[.1]*3},{'closure_loads':[0.]*4},{'closure_loads':[4.]*4},
    {'closure_loads':[float('nan')]*4},{'probe_channel':'left_thumb'},
])
def test_incomplete_or_conflicting_sequence_rejected(change):
    with pytest.raises(ValueError):validate_continuous_options(options(**change))


def test_legacy_options_not_changed():
    validate_continuous_options(options(grasp_sequence=False,closure_pilot=True,approach_certificate=None))


def test_runner_contains_measured_sequence_and_joint_bimanual_publication():
    source=(Path(__file__).parents[1]/'scripts/verify_fixed_dual_panda_dh116.py').read_text()
    tree=ast.parse(source)
    calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
    assert '--grasp-sequence' in source
    assert 'validate_continuous_options' in calls
    assert 'TabletopGraspSequence' in calls
    assert 'grasp_sequence.update' in calls
    assert 'coordinated_bimanual_increment' in calls
    assert 'grasp_sequence.transport_velocity' in calls
    assert "record['grasp_sequence']" in source
    assert "result['continuous_grasp']" in source
    assert 'permit_pad_contact=closure_active' in source


def test_failed_final_probe_cannot_leave_positive_nested_grasp_claim():
    source=(Path(__file__).parents[1]/'scripts/verify_fixed_dual_panda_dh116.py').read_text()
    tree=ast.parse(source);scope=next(n for n in tree.body if isinstance(n,ast.Try))
    cleanup='\n'.join(ast.unparse(n) for n in scope.finalbody)
    assert "if not result['passed']:" in cleanup
    assert "result[key]['grasp_qualified'] = False" in cleanup
    assert "result['continuous_grasp']['grasp_lift_passed'] = False" in cleanup


def test_sequence_uses_actual_station_height_and_retains_approach_transition():
    source=(Path(__file__).parents[1]/'scripts/verify_fixed_dual_panda_dh116.py').read_text()
    assert "nodes=np.asarray(snapshot['rod_position']).T" in source
    assert 'radius=np.asarray(engine.rod.radius)' in source
    assert 'station_heights_m=station_heights' in source
    assert "previous_stage=='approach' and evidence['stage']=='close'" in source
    assert 'if approach is not None and not args.grasp_sequence:' in source


def test_final_phase_summary_is_computed_after_probe_qualification():
    source=(Path(__file__).parents[1]/'scripts/verify_fixed_dual_panda_dh116.py').read_text()
    cleanup=source.split('\nfinally:\n',1)[1]
    assert "result['stage_status']" in cleanup
    assert cleanup.index('grasp_sequence.stage_summary')>cleanup.index("result['passed']=result['passed'] and result['point_probe']['passed']")
