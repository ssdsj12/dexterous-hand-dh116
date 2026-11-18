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
        steps=200000,closure_loads=[.87,.77,.50,.88,.93,.50],probe_channel=None)
    values.update(changes);return SimpleNamespace(**values)


def test_valid_continuous_options():
    validate_continuous_options(options())


def test_touch_first_load_floor_checked_before_isaac_launch():
    with pytest.raises(ValueError):
        validate_continuous_options(options(closure_loads=[.049]*6))
    validate_continuous_options(options(closure_loads=[.05]*6))


@pytest.mark.parametrize('change',[
    {'closure_pilot':True},{'active_pose':None},{'active_pose_sha256':None},
    {'approach_certificate':None},{'approach_certificate_sha256':None},
    {'point_references':None},{'contact_manifest':None},{'contact_manifest_sha256':None},
    {'disable_contact_reaction':True},{'coordinated_closure':False},
    {'gravity_hold':False},{'table_wire':False},{'physics_dt':.001},
    {'physics_dt':float('nan')},{'steps':30000},{'closure_loads':None},
    {'closure_loads':[.1]*3},{'closure_loads':[.1]*4},{'closure_loads':[0.]*6},{'closure_loads':[4.]*6},
    {'closure_loads':[float('nan')]*6},{'probe_channel':'left_thumb'},
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
    assert 'permit_pad_contact=bool(closure_active or approach_pad_contact)' in source
    assert "approach_pad_contact=bool(grasp_sequence is not None and grasp_sequence.stage=='approach')" in source


def test_hand_mpc_mode_preserves_full_scene_and_separates_action_owners():
    source=(Path(__file__).parents[1]/'scripts/verify_fixed_dual_panda_dh116.py').read_text()
    tree=ast.parse(source)
    calls=[ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node,ast.Call)]
    assert "parser.add_argument('--hand-mpc'" in source
    assert "parser.add_argument('--hand-mpc-update-steps'" in source
    assert "spawn.func('/World/Rig',spawn)" in source
    assert 'fixed_dual_panda_dh116.usd' in source
    assert 'IntegratedHandMPC' in source
    assert 'ParallelContactMPC' in source
    assert 'wire_surface_targets' in calls
    assert 'route_phase_targets' in calls
    assert 'panda_feedback_increment' in calls
    assert "hand_targets[side]" in source
    assert "target.update(zip([f'{side}_panda_joint{i}'" in source
    assert "hand_mpc_action_dim" in source


def test_hand_mpc_requires_the_existing_guarded_grasp_sequence():
    source=(Path(__file__).parents[1]/'scripts/verify_fixed_dual_panda_dh116.py').read_text()
    assert 'args.hand_mpc and not args.grasp_sequence' in source
    assert 'mpc_action_dim=12' in source


def test_contact_mpc_reference_uses_measured_load_and_gap_feedback():
    source=(Path(__file__).parents[1]/'scripts/verify_fixed_dual_panda_dh116.py').read_text()
    assert 'GripLoadReference' in source
    assert 'grip_load_reference.update' in source
    assert 'penetration_m=mpc_penetration' in source
    assert 'contact_loads_n=previous_loads.tolist()' in source


def test_hand_mpc_uses_rate_limited_adaptive_force_targets():
    source=(Path(__file__).parents[1]/'scripts/verify_fixed_dual_panda_dh116.py').read_text()
    assert 'AdaptiveForceController' in source
    assert 'adaptive_force_controller.update' in source
    assert 'adaptive_target_loads_n' in source


def test_runner_records_measured_lift_failure_class_for_recovery():
    source=(Path(__file__).parents[1]/'scripts/verify_fixed_dual_panda_dh116.py').read_text()
    assert 'diagnose_lift' in source
    assert "record['recovery']" in source


def test_direct_close_passes_candidate_pose_reference_to_hand_mpc():
    source=(Path(__file__).parents[1]/'scripts/verify_fixed_dual_panda_dh116.py').read_text()
    assert 'load_pose_lock' in source
    assert 'reference_active_joints=' in source
    assert 'reference_tolerance_rad=' in source


def test_lift_routes_measured_palm_feedback_through_panda_controller():
    source=(Path(__file__).parents[1]/'scripts/verify_fixed_dual_panda_dh116.py').read_text()
    assert 'panda_feedback_increment' in source
    assert 'palm_target_positions_w=' in source
    assert 'palm_actual_positions_w=' in source
    assert "record['lift_tracking']" in source


def test_candidate_pose_direct_close_rejects_imported_approach_certificate():
    source=(Path(__file__).parents[1]/'scripts/verify_fixed_dual_panda_dh116.py').read_text()
    assert "args.grasp_entry=='close' and args.active_pose and args.approach_certificate" in source


def test_contact_mpc_receives_actual_nonpad_hand_clearance_constraints():
    source=(Path(__file__).parents[1]/'scripts/verify_fixed_dual_panda_dh116.py').read_text()
    assert 'hand_wire_clearance_observations' in source
    assert 'clearance_body_names=' in source
    assert "path in source_scene.pad_bodies" in source
    assert "clearance_jacobians=mpc_clearance['clearance_jacobians']" in source


def test_direct_close_can_initialize_measured_robot_and_gpu_rod_without_phase_credit():
    source=(Path(__file__).parents[1]/'scripts/verify_fixed_dual_panda_dh116.py').read_text()
    assert "parser.add_argument('--close-initial-state'" in source
    assert 'load_close_initialization' in source
    assert "engine.restore_physical_state(close_initial_payload['rod'])" in source
    assert 'robot_initial_arrays' in source
    assert 'imported_phase_evidence=False' in source


def test_hand_mpc_can_load_student_checkpoint_without_panda_observation():
    source=(Path(__file__).parents[1]/'scripts/verify_fixed_dual_panda_dh116.py').read_text()
    assert "parser.add_argument('--hand-policy'" in source
    assert 'load_mounted_policy' in source
    assert "parser.add_argument('--hand-policy-role'" in source
    assert 'hand_mpc_controller.plan' in source
    assert 'hand_mpc_previous_action' in source
    assert 'hand_mpc_grasp_progress' in source
    assert "result['hand_mpc_action_dim']=12" in source


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
    assert "radius=(engine.radius.detach().cpu().numpy() if args.rod_backend=='torch_cuda'" in source
    assert 'else np.asarray(engine.rod.radius))' in source
    assert 'station_heights_m=station_heights' in source
    assert "previous_stage=='approach' and evidence['stage']=='close'" in source
    assert 'if approach is not None and not args.grasp_sequence:' in source


def test_final_phase_summary_is_computed_after_probe_qualification():
    source=(Path(__file__).parents[1]/'scripts/verify_fixed_dual_panda_dh116.py').read_text()
    cleanup=source.split('\nfinally:\n',1)[1]
    assert "result['stage_status']" in cleanup
    assert cleanup.index('grasp_sequence.stage_summary')>cleanup.index("result['passed']=result['passed'] and result['point_probe']['passed']")
