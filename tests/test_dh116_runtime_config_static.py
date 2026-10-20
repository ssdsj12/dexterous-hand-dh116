"""CLI wiring contract without starting Isaac during unit tests."""
import ast


def test_motion_value_and_prediction_source_is_in_backend_identity():
    source=(Path(__file__).parents[1]/'scripts/verify_fixed_dual_panda_dh116.py').read_text()
    tree=ast.parse(source)
    assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and
        any(ast.unparse(t)=="result['backend_pins']" for t in n.targets))
    values=[n.value for n in ast.walk(assignment.value.generators[0].iter)
            if isinstance(n,ast.Constant) and isinstance(n.value,str)]
    assert 'co_sim/moving_contact_models.py' in values


def test_coordinated_closure_is_explicit_uses_measured_palm_and_updates_both_targets():
    source=(Path(__file__).parents[1]/'scripts/verify_fixed_dual_panda_dh116.py').read_text()
    tree=ast.parse(source)
    option=next((n for n in ast.walk(tree) if isinstance(n,ast.Call) and
                 isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument' and
                 n.args and isinstance(n.args[0],ast.Constant) and
                 n.args[0].value=='--coordinated-closure'),None)
    assert option is not None
    assert any(k.arg=='action' and ast.literal_eval(k.value)=='store_true' for k in option.keywords)
    assert 'grasp_mode=args.closure_pilot or args.grasp_sequence' in source
    assert 'if args.coordinated_closure and not grasp_mode:' in source
    assert 'coordinated_closure=args.coordinated_closure' in source
    assert "arm.body_names.index('base_link')" in source
    assert 'raw[pi-1,3:][:,arm_ids]' in source
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and
           isinstance(n.func,ast.Name) and n.func.id=='coordinated_closure_increment']
    assert len(calls)==1
    assert [ast.unparse(x) for x in calls[0].args[:3]]==['point_rows_control','normals','palm_angular']
    assert 'closure_arm_targets[side]+=increment[:7]' in source
    assert 'closure_targets[side][:3]+=increment[7:]' in source
    assert 'target_arms[side]=closure_arm_targets[side]' in source


def test_closure_failure_preserves_candidate_rod_state_before_reaction_publish():
    source=(Path(__file__).parents[1]/'scripts/verify_fixed_dual_panda_dh116.py').read_text()
    tree=ast.parse(source)
    assignments=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and
                 any(ast.unparse(t)=="record['rod_state']" for t in n.targets)]
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='bridge.publish']
    assert len(assignments)==1 and len(calls)==1
    assert assignments[0].lineno<calls[0].lineno
    assert "record['contact_candidate']" in source


def test_closure_material_progress_is_printed_without_waiting_for_shutdown():
    source=(Path(__file__).parents[1]/'scripts/verify_fixed_dual_panda_dh116.py').read_text()
    calls=[n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Call) and
           isinstance(n.func,ast.Name) and n.func.id=='print']
    messages=[n for n in calls if n.args and isinstance(n.args[0],ast.Constant) and
              n.args[0].value=='DH116_MATERIAL']
    assert len(messages)==1
    assert "record['material_observation']" in ast.unparse(messages[0])
    assert any(k.arg=='flush' and isinstance(k.value,ast.Constant) and k.value.value is True
               for k in messages[0].keywords)


def test_contact_broad_phase_is_explicit_shared_and_pins_mesh_source():
    source=(Path(__file__).parents[1]/'scripts/verify_fixed_dual_panda_dh116.py').read_text()
    assert '--contact-broad-phase' in source
    calls=[n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Call) and
           isinstance(n.func,ast.Name) and n.func.id in ('MovingContactRodEngine','TableRodEngine')]
    for call in calls:
        assert any(k.arg=='contact_broad_phase' and ast.unparse(k.value)=='args.contact_broad_phase' for k in call.keywords)
    assert 'contact_broad_phase=args.contact_broad_phase' in source
    assert 'elastica/rigidbody/mesh_rigid_body.py' in source
from pathlib import Path


def test_clearance_certificate_cache_is_explicit_and_recorded():
    source=(Path(__file__).parents[1]/'scripts/verify_fixed_dual_panda_dh116.py').read_text()
    assert '--clearance-certificate-cache' in source
    tree=ast.parse(source)
    scene=next(n for n in ast.walk(tree) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Name) and n.func.id=='SourceCollisionScene')
    assert any(k.arg=='cache_certificates' and ast.unparse(k.value)=='args.clearance_certificate_cache'
        for k in scene.keywords)
    assert 'clearance_certificate_cache=args.clearance_certificate_cache' in source


def test_tangential_regularization_is_explicit_and_shared_by_backend_instances():
    source=(Path(__file__).parents[1]/'scripts/verify_fixed_dual_panda_dh116.py').read_text()
    tree=ast.parse(source)
    assert '--contact-tangential-damping' in source
    constructors=[n for n in ast.walk(tree) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Name) and n.func.id in ('MovingContactRodEngine','TableRodEngine')]
    for call in constructors:
        assert any(k.arg=='velocity_damping_coefficient' and ast.unparse(k.value)=='args.contact_tangential_damping'
                   for k in call.keywords)
    assert 'contact_tangential_damping=args.contact_tangential_damping' in source


def test_fixed_material_observation_is_after_actual_physics_readback():
    source=(Path(__file__).parents[1]/'scripts/verify_fixed_dual_panda_dh116.py').read_text()
    tree=ast.parse(source)
    loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.target)=='step')
    calls=[n for n in ast.walk(loop) if isinstance(n,ast.Call)]
    physics=next(n.lineno for n in calls if ast.unparse(n.func)=='sim.step')
    anchors=next(n.lineno for n in calls if ast.unparse(n.func)=='material_in_palm')
    assert anchors>physics
    assert "record['material_observation']" in source
    assert 'material_ids=fixed_material_ids' in source
    assert 'diagnostic_contact_hold_not_grasp' in source


def test_contact_quadrature_is_explicit_and_shared_with_observation():
    source=(Path(__file__).parents[1]/'scripts/verify_fixed_dual_panda_dh116.py').read_text()
    tree=ast.parse(source)
    assert '--contact-subsamples' in source
    constructors=[n for n in ast.walk(tree) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Name) and n.func.id in ('MovingContactRodEngine','TableRodEngine')]
    assert len(constructors)>=2
    for call in constructors:
        assert any(k.arg=='contact_subsamples' and ast.unparse(k.value)=='args.contact_subsamples'
                   for k in call.keywords)
    assert 'engine.contact_geometry(c)' in source
    assert 'element_ids=' in source


def test_hand_effort_is_explicit_and_shared_by_drive_and_gravity():
    path=Path(__file__).parents[1]/'scripts/verify_fixed_dual_panda_dh116.py'
    tree=ast.parse(path.read_text());calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    assert any(isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument'
        and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value=='--hand-effort-limit' for n in calls)
    gravity=[n for n in calls if isinstance(n.func,ast.Name) and n.func.id=='compensated_efforts']
    assert len(gravity)==1 and ast.unparse(gravity[0].args[-1])=='args.hand_effort_limit'
    hand=[n for n in calls if isinstance(n.func,ast.Name) and n.func.id=='ImplicitActuatorCfg'
        and any(k.arg=='joint_names_expr' and ast.unparse(k.value)=='list(ACTIVE)' for k in n.keywords)]
    assert len(hand)==1
    assert ast.unparse(next(k.value for k in hand[0].keywords if k.arg=='effort_limit_sim'))=='args.hand_effort_limit'


def test_approach_uses_measured_source_guard_before_physics_and_after_readback():
    path=Path(__file__).parents[1]/'scripts/verify_fixed_dual_panda_dh116.py'
    tree=ast.parse(path.read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    assert any(isinstance(n.func,ast.Name) and n.func.id=='load_approach' for n in calls)
    checks=[n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='check'
            and any(k.arg=='measured_body_poses' for k in n.keywords)]
    assert checks, 'nominal FK is not a measured collision guard'
    loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.target)=='step')
    loopcalls=[n for n in ast.walk(loop) if isinstance(n,ast.Call)]
    guards=sorted(n.lineno for n in loopcalls if isinstance(n.func,ast.Name) and n.func.id=='measured_source_guard')
    physics=next(n.lineno for n in loopcalls if ast.unparse(n.func)=='sim.step')
    assert len(guards)>=2 and guards[0]<physics<guards[-1]


def test_failure_path_keeps_last_step_and_partial_trajectory():
    path=Path(__file__).parents[1]/'scripts/verify_fixed_dual_panda_dh116.py'
    tree=ast.parse(path.read_text())
    scope=next(n for n in tree.body if isinstance(n,ast.Try))
    handler='\n'.join(ast.unparse(n) for n in scope.handlers)
    cleanup='\n'.join(ast.unparse(n) for n in scope.finalbody)
    assert "result['failure_step']" in handler
    assert "result['last_attempted_record']" in handler
    # Completed rows are flushed during the loop, not rewritten on failure.
    # Runtime journal tests exercise preservation on an actual exception.
    body='\n'.join(ast.unparse(n) for n in scope.body)
    assert "rows = EvidenceJournal(args.output / 'trajectory.jsonl')" in body
    assert 'rows.append(record)' in body
    assert 'rows.close()' in cleanup
    assert "open('w')" not in cleanup


def test_failed_physics_trial_also_preserves_completed_point_jacobian_probes():
    path=Path(__file__).parents[1]/'scripts/verify_fixed_dual_panda_dh116.py'
    tree=ast.parse(path.read_text())
    scope=next(n for n in tree.body if isinstance(n,ast.Try))
    cleanup='\n'.join(ast.unparse(n) for n in scope.finalbody)
    assert 'point_probe.json' in cleanup
    assert "result['point_probe']" in cleanup
    assert "result['passed'] and result['point_probe']['passed']" in cleanup


def test_closure_pilot_is_separate_and_keeps_measured_point_jacobian_and_region_guards():
    path=Path(__file__).parents[1]/'scripts/verify_fixed_dual_panda_dh116.py'
    source=path.read_text();tree=ast.parse(source)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    assert '--closure-pilot' in source
    for required in ('closure_increment','point_jacobian','normal_contact_sample','PadRegionGuard'):
        assert any(isinstance(n.func,ast.Name) and n.func.id==required for n in calls),required
    assert 'diagnostic_initial_touch_not_continuous_grasp' in source
    assert 'rod_path_bound_m' in source and 'physical_motion_delta' in source
    predicate=ast.dump(ast.parse('not args.probe_channel and not grasp_mode',mode='eval').body)
    assert any(isinstance(n,ast.If) and ast.dump(n.test)==predicate
        for n in ast.walk(tree)), 'loaded closure is not a stationary wire-settling test'
