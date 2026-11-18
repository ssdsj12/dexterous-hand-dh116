"""Synthetic CPU arithmetic/contracts only; no Native or prediction qualification."""
from dataclasses import replace
import importlib
import importlib.util
import math

import pytest

from go2_pvcnn.control.dh116.tabletop_bimanual_mpc_state import SampleClock


MODULE = 'go2_pvcnn.control.dh116.tabletop_bimanual_mpc_objective'
TERMS = ('geometry', 'contact', 'support', 'slip', 'bimanual', 'joint', 'smooth', 'risk')
SHAPES = dict(geometry=(8, 2, 3), contact=(8, 2, 3), support=(1, 2),
              slip=(1, 2, 3), bimanual=(8, 6), joint=(8, 2, 11),
              collision=(8, 2), overload=(8, 2, 3), tracking=(8, 2, 11))
UNITS = dict(geometry='m', contact='dimensionless', support='N', slip='m/s',
             bimanual='m|rad', joint='joint_coordinate_per_dof', collision='dimensionless',
             overload='dimensionless', tracking='dimensionless')


def api():
    assert importlib.util.find_spec(MODULE) is not None, 'eight-term objective is missing'
    return importlib.import_module(MODULE)


def clock(**changes):
    base = SampleClock('synthetic-run', 0, 'episode-a', 480, 1., 960, 2., 60, 3)
    return replace(base, **changes)


def filled(shape, value):
    return [filled(shape[1:], value) for _ in range(shape[0])] if shape else value


def make_input(**changes):
    m = api()
    c = clock()
    actions = [[.1 * (t + 1)] * 24 for t in range(8)]
    digest = m.candidate_digest(actions)
    values = dict(geometry=.2, contact=.3, support=2., slip=.4, bimanual=.1,
                  joint=.2, collision=.5, overload=.6, tracking=.7)
    evidence = {}
    for name, shape in SHAPES.items():
        measured = name in ('support', 'slip')
        evidence[name] = m.ResidualEvidence(
            values=filled(shape, values[name]), valid_mask=filled(shape, True),
            source=f'synthetic:{name}:not_native', unit=UNITS[name],
            frame=('world' if name in ('geometry', 'support', 'slip') else
                   'relative_left_grasp' if name == 'bimanual' else
                   'joint' if name in ('joint', 'tracking') else 'declared_residual'),
            clock=c, scope='measured_current' if measured else 'predicted_rollout',
            qualification='cpu_fixture_only',
            candidate_digest=None if measured else digest,
            contact_basis='accepted_contact' if name == 'contact' else 'not_applicable',
            contact_mask=filled(shape, True) if name == 'contact' else None,
            control_period_s=None if measured else 1 / 60,
            coordinate_units=('rad',) * 11 if name == 'joint' else None)
    args = dict(clock=c, current_clock=c, max_age_s=.01, control_period_s=1 / 60, actions=actions,
                previous_action=[0.] * 24, previous_action_clock=c,
                previous_action_source='synthetic:last_actually_issued24',
                active_hands=[True, True], active_pads=filled((2, 3), True),
                required_support_N=7.,
                bimanual_translation_scale_m=.2, bimanual_rotation_scale_rad=.5,
                joint_dof_names=tuple(f'finger_{i}' for i in range(11)),
                joint_scope_source='synthetic:finger_only_fixture_subset',
                joint_coordinate_units=('rad',) * 11, joint_reference_per_dof=(.1,) * 11,
                joint_reference_source='synthetic:experimental_rad_reference_not_physical_limit',
                residuals=evidence)
    args.update(changes)
    return m.ObjectiveInput(**args)


def weights(**changes):
    return api().ObjectiveWeights(**{name: changes.get(name, i + 1.)
                                    for i, name in enumerate(TERMS)})


def evaluate(data=None, w=None, active=TERMS):
    return api().evaluate_objective(data or make_input(), w or weights(), active_costs=active)


def test_every_raw_weighted_and_total_is_independently_recomputable():
    result = evaluate()
    expected = dict(geometry=8 * 2 * 3 * .2**2,
                    contact=8 * 2 * 3 * .3,
                    support=(7 - (2 + 2))**2,
                    slip=2 * 3 * .4**2,
                    bimanual=8 * (3 * (.1 / .2)**2 + 3 * (.1 / .5)**2),
                    joint=8 * 2 * 11 * (.1 / .2)**2,
                    smooth=8 * 24 * .1**2,
                    risk=8 * 2 * .5**2 + 8 * 2 * 3 * .6**2 + 8 * 2 * 11 * .7**2)
    for i, name in enumerate(TERMS):
        term = result.terms[name]
        assert term.available
        assert term.missing_reason is None
        assert term.raw == pytest.approx(expected[name])
        assert term.weight == i + 1
        assert term.weighted == pytest.approx(expected[name] * (i + 1))
        assert term.source and term.raw_unit and term.input_units and term.qualification
    assert result.total == pytest.approx(sum(expected[n] * (i + 1) for i, n in enumerate(TERMS)))
    assert result.full_spec_cost_valid
    assert not result.native_qualified
    assert not result.prediction_validated
    assert not result.solve_qualified
    assert not result.action_admissible


def test_zero_weight_does_not_turn_missing_support_into_valid_zero():
    data = make_input()
    missing = replace(data, residuals={k: v for k, v in data.residuals.items() if k != 'support'})
    report = evaluate(missing, weights(support=0.))
    term = report.terms['support']
    assert term.raw is None and term.weighted is None and not term.available
    assert term.missing_reason and not report.full_spec_cost_valid
    assert report.total is None
    subset = evaluate(missing, weights(support=0.), active=('geometry', 'smooth'))
    assert subset.total == pytest.approx(subset.terms['geometry'].weighted + subset.terms['smooth'].weighted)
    assert not subset.full_spec_cost_valid
    zero = make_input(required_support_N=1.)
    assert evaluate(zero).terms['support'].raw == 0.


def test_current_support_and_slip_are_single_sample_constants_when_actions_change():
    original = make_input()
    actions = [[.9] * 24 for _ in range(8)]
    # Only a measured subset is evaluated: other rollout evidence belongs to old U.
    residuals = {k: v for k, v in original.residuals.items() if k in ('support', 'slip')}
    a = evaluate(replace(original, residuals=residuals), active=('support', 'slip', 'smooth'))
    b = evaluate(replace(original, actions=actions, residuals=residuals), active=('support', 'slip', 'smooth'))
    for name in ('support', 'slip'):
        assert a.terms[name].raw == b.terms[name].raw
        assert b.terms[name].optimization_effect == 'constant_wrt_candidate'
        assert b.terms[name].prediction_scope == ('measured_current',)
    assert a.terms['smooth'].raw != b.terms['smooth'].raw
    assert b.terms['smooth'].optimization_effect == 'candidate_dependent'
    assert not b.prediction_validated


def test_geometry_only_cannot_supply_actual_contact_or_support():
    data = make_input()
    residuals = dict(data.residuals)
    residuals['contact'] = replace(residuals['contact'], contact_basis='geometry_only')
    del residuals['support']
    report = evaluate(replace(data, residuals=residuals), active=('geometry',))
    assert report.terms['geometry'].available
    assert not report.terms['contact'].available
    assert 'contact basis' in report.terms['contact'].missing_reason
    assert not report.terms['support'].available
    residuals['contact'] = replace(data.residuals['contact'], contact_mask=filled((8, 2, 3), False))
    assert not evaluate(replace(data, residuals=residuals), active=('geometry',)).terms['contact'].available


def test_missing_pad_remains_none_without_losing_valid_sibling_or_other_env():
    data = make_input()
    partial = filled((8, 2, 3), None)
    mask = filled((8, 2, 3), False)
    for t in range(8):
        partial[t][0][0], mask[t][0][0] = .2, True
    residuals = dict(data.residuals)
    residuals['geometry'] = replace(residuals['geometry'], values=partial,
                                    valid_mask=mask, missing_reason='other pads missing')
    report = evaluate(replace(data, residuals=residuals), active=('smooth',))
    term = report.terms['geometry']
    assert term.raw is None and not term.available
    assert term.observed_raw == pytest.approx(8 * .2**2)
    assert term.element_costs['geometry'][0][0][0] == pytest.approx(.2**2)
    assert term.element_costs['geometry'][0][1][2] is None
    selected = replace(data, residuals=residuals, active_hands=[True, False],
                       active_pads=[[True, False, False], [False, False, False]])
    assert evaluate(selected, active=('geometry',)).terms['geometry'].raw == pytest.approx(8 * .2**2)
    env1 = clock(env_id=1, episode_uid='env1-episode')
    with pytest.raises(ValueError, match='clock'):
        evaluate(replace(selected, residuals={'geometry': replace(residuals['geometry'], clock=env1)}))


def test_risk_component_missing_is_not_hidden_by_other_available_components():
    data = make_input()
    report = evaluate(replace(data, residuals={k: v for k, v in data.residuals.items() if k != 'overload'}),
                      active=('smooth',))
    assert report.terms['risk'].raw is None
    assert not report.terms['risk'].available
    assert report.terms['risk'].observed_raw == pytest.approx(16 * .5**2 + 176 * .7**2)
    assert 'overload' in report.terms['risk'].missing_reason


def test_partial_support_cannot_treat_unobserved_other_hand_as_zero():
    data = make_input()
    residuals = dict(data.residuals)
    residuals['support'] = replace(residuals['support'], values=[[2., None]],
                                  valid_mask=[[True, False]], missing_reason='right support NA')
    report = evaluate(replace(data, residuals=residuals), active=('smooth',))
    assert report.terms['support'].raw is None
    assert report.terms['support'].observed_raw is None


@pytest.mark.parametrize('name,value', [('geometry', -1.), ('support', math.nan), ('risk', math.inf), ('joint', True)])
def test_invalid_weights_rejected(name, value):
    with pytest.raises(ValueError, match='weight'):
        weights(**{name: value})


@pytest.mark.parametrize('shape', [(7, 24), (8, 23), (8, 2, 12)])
def test_actions_are_exactly_eight_control_periods_by_canonical24(shape):
    with pytest.raises(ValueError, match='actions.*shape'):
        make_input(actions=filled(shape, 0.))


@pytest.mark.parametrize('bad', [math.nan, math.inf, True])
def test_nonfinite_or_boolean_action_and_residual_rejected(bad):
    actions = filled((8, 24), 0.)
    actions[0][0] = bad
    with pytest.raises(ValueError, match='finite'):
        make_input(actions=actions)
    data = make_input()
    values = filled((8, 2, 3), .2)
    values[0][0][0] = bad
    with pytest.raises(ValueError, match='finite'):
        evaluate(replace(data, residuals={'geometry': replace(data.residuals['geometry'], values=values)}))


@pytest.mark.parametrize('change', [dict(terminal=True), dict(reset=True),
                                    dict(env_id=1), dict(episode_uid='new'), dict(reset_generation=4),
                                    dict(world_step=500, world_time_s=1.1,
                                         sdk_step=980, sdk_time_s=2.1)])
def test_terminal_reset_stale_and_cross_identity_never_reused(change):
    with pytest.raises(ValueError):
        make_input(current_clock=clock(**change))


def test_future_rollout_bound_to_candidate_and_requires_horizon8():
    data = make_input()
    actions = filled((8, 24), .8)
    with pytest.raises(ValueError, match='candidate'):
        evaluate(replace(data, actions=actions))
    bad = replace(data.residuals['geometry'], values=filled((1, 2, 3), .2), valid_mask=filled((1, 2, 3), True))
    with pytest.raises(ValueError, match='shape'):
        evaluate(replace(data, residuals={'geometry': bad}))
    measured = replace(data.residuals['slip'], values=filled((8, 2, 3), .4), valid_mask=filled((8, 2, 3), True))
    with pytest.raises(ValueError, match='shape'):
        evaluate(replace(data, residuals={'slip': measured}))


def test_units_missing_reason_masks_and_joint_barrier_domain_are_strict():
    data = make_input()
    with pytest.raises(ValueError, match='unit'):
        evaluate(replace(data, residuals={'geometry': replace(data.residuals['geometry'], unit='mm')}))
    with pytest.raises(ValueError, match='None'):
        bad = replace(data.residuals['geometry'], valid_mask=filled((8, 2, 3), False), missing_reason='NA')
        evaluate(replace(data, residuals={'geometry': bad}))
    with pytest.raises(ValueError, match='positive'):
        bad = replace(data.residuals['joint'], values=filled((8, 2, 11), 0.))
        evaluate(replace(data, residuals={'joint': bad}))
    with pytest.raises(ValueError, match='previous_action.*shape'):
        make_input(previous_action=[0.] * 23)


def test_inputs_and_reports_detach_mutable_callers_and_expose_no_control_output():
    data = make_input()
    external = filled((8, 24), .8)
    isolated = replace(data, actions=external, residuals={})
    external[0][0] = 100.
    assert isolated.actions[0][0] == .8
    result = evaluate(isolated, active=('smooth',))
    with pytest.raises(TypeError):
        result.terms['geometry'] = result.terms['smooth']
    assert not hasattr(result, 'action') and not hasattr(result, 'success')
    assert not hasattr(api(), 'solve')


def test_joint_cost_names_variable_articulation_axes_including_wrist():
    data = mixed_joint_input()
    report = evaluate(data)
    assert report.joint_dof_names == data.joint_dof_names
    assert len(report.joint_dof_names) == 17
    assert len(report.terms['joint'].element_costs['joint'][0][0]) == 17
    assert data.joint_scope_source in report.terms['joint'].source
    assert not report.action_admissible


def test_declared_future_support_is_candidate_dependent_but_never_validated():
    data = make_input()
    predicted = replace(data.residuals['support'], values=filled((8, 2), 2.),
                        valid_mask=filled((8, 2), True), scope='predicted_rollout',
                        candidate_digest=api().candidate_digest(data.actions),
                        control_period_s=data.control_period_s,
                        source='synthetic:unvalidated_contact_predictor')
    report = evaluate(replace(data, residuals={'support': predicted}), active=('support',))
    assert report.terms['support'].raw == pytest.approx(8 * 3**2)
    assert report.terms['support'].optimization_effect == 'candidate_dependent'
    assert not report.prediction_validated and not report.solve_qualified


def test_weight_change_changes_corresponding_candidate_ranking_without_solving():
    data = make_input()
    alternative = replace(data, actions=filled((8, 24), 0.), residuals={})
    geometry = replace(data.residuals['geometry'], values=filled((8, 2, 3), .4),
                       candidate_digest=api().candidate_digest(alternative.actions))
    alternative = replace(alternative, residuals={'geometry': geometry})
    weak = weights(geometry=.01, smooth=100.)
    strong = weights(geometry=100., smooth=.01)
    assert evaluate(data, weak, active=('geometry', 'smooth')).total > evaluate(alternative, weak, active=('geometry', 'smooth')).total
    assert evaluate(data, strong, active=('geometry', 'smooth')).total < evaluate(alternative, strong, active=('geometry', 'smooth')).total


def test_prediction_time_grid_cannot_be_reinterpreted_as_physics_substeps():
    data = make_input()
    with pytest.raises(ValueError, match='control period'):
        evaluate(replace(data, control_period_s=1 / 480))


@pytest.mark.parametrize('wrong_period', [1 / 480, 1 / 30, .1])
def test_matching_wrong_input_and_every_prediction_period_are_rejected(wrong_period):
    data = make_input()
    with pytest.raises(ValueError, match='60.*control period|control period.*60'):
        same_wrong_grid = {name: replace(evidence, control_period_s=wrong_period)
                           if evidence.scope == 'predicted_rollout' else evidence
                           for name, evidence in data.residuals.items()}
        evaluate(replace(data, control_period_s=wrong_period, residuals=same_wrong_grid))


def test_fixed_control_period_also_applies_without_any_future_residual():
    with pytest.raises(ValueError, match='60.*control period|control period.*60'):
        make_input(control_period_s=1 / 480, residuals={})


def test_separate_envs_keep_their_own_hand_pad_masks_without_normalizing_by_others():
    data = make_input()
    def one_pad(c, hand, pad):
        values, mask = filled((8, 2, 3), None), filled((8, 2, 3), False)
        pads = filled((2, 3), False)
        pads[hand][pad] = True
        for t in range(8):
            values[t][hand][pad], mask[t][hand][pad] = .2, True
        evidence = replace(data.residuals['geometry'], values=values, valid_mask=mask,
                           clock=c, missing_reason='other pads not measured')
        return replace(data, clock=c, current_clock=c, previous_action_clock=c,
                       active_pads=pads, residuals={'geometry': evidence})
    a = evaluate(one_pad(clock(), 0, 0), active=('geometry',))
    b = evaluate(one_pad(clock(env_id=1, episode_uid='separate'), 1, 2), active=('geometry',))
    assert a.terms['geometry'].raw == b.terms['geometry'].raw == pytest.approx(8 * .2**2)
    assert a.terms['geometry'].element_costs['geometry'][0][1][2] is None
    assert b.terms['geometry'].element_costs['geometry'][0][0][0] is None


def test_extreme_finite_values_fail_closed_instead_of_returning_nonfinite_costs():
    data = make_input()
    bad = replace(data.residuals['geometry'], values=filled((8, 2, 3), 1e308))
    with pytest.raises(ValueError, match='finite'):
        evaluate(replace(data, residuals={'geometry': bad}), active=('geometry',))


def test_explicit_unavailable_evidence_keeps_reason_source_and_false_mask():
    data = make_input()
    missing = replace(data.residuals['support'], values=None, valid_mask=filled((1, 2), False),
                      missing_reason='qualified accepted-pad support not exposed')
    report = evaluate(replace(data, residuals={'support': missing}), active=('support',))
    term = report.terms['support']
    assert term.raw is None and term.weighted is None and report.total is None
    assert term.source == (missing.source,)
    assert term.valid_mask['support'] == ((False, False),)
    assert missing.missing_reason in term.missing_reason


def mixed_joint_input():
    data = make_input()
    names = ('wrist_x', 'wrist_y', 'wrist_z', 'wrist_roll', 'wrist_pitch', 'wrist_yaw') + tuple(
        f'finger_{i}' for i in range(11))
    units = ('m',) * 3 + ('rad',) * 14
    reference = (.01, .02, .04) + (.1,) * 14
    left_margins = (.02, .08, .16) + (.2,) * 14
    right_margins = (.04, .04, .08) + (.4,) * 14
    residuals = dict(data.residuals)
    residuals['joint'] = replace(residuals['joint'], unit='joint_coordinate_per_dof',
                                coordinate_units=units,
                                values=[[left_margins, right_margins] for _ in range(8)],
                                valid_mask=filled((8, 2, 17), True))
    residuals['tracking'] = replace(residuals['tracking'], values=filled((8, 2, 17), .2),
                                   valid_mask=filled((8, 2, 17), True))
    return replace(data, joint_dof_names=names, joint_coordinate_units=units,
                   joint_reference_per_dof=reference,
                   joint_reference_source='synthetic:experimental_reference_with_m_and_rad',
                   joint_scope_source='synthetic:mounted17_axis_order_and_SI_units', residuals=residuals)


def test_joint_barrier_uses_same_unit_per_axis_for_prismatic_and_revolute_dofs():
    data = mixed_joint_input()
    report = evaluate(data)
    # First three are physical m, the remaining fourteen are physical rad.
    left = (.01 / .02)**2 + (.02 / .08)**2 + (.04 / .16)**2 + 14 * (.1 / .2)**2
    right = (.01 / .04)**2 + (.02 / .04)**2 + (.04 / .08)**2 + 14 * (.1 / .4)**2
    term = report.terms['joint']
    assert term.raw == pytest.approx(8 * (left + right))
    assert term.raw_unit == 'dimensionless'
    assert term.element_costs['joint'][0][0][0] == pytest.approx((.01 / .02)**2)
    assert term.element_costs['joint'][0][1][3] == pytest.approx((.1 / .4)**2)
    assert report.joint_coordinate_units == ('m',) * 3 + ('rad',) * 14
    assert report.joint_reference_per_dof == data.joint_reference_per_dof
    assert data.joint_reference_source in term.source
    assert not report.action_admissible and not report.prediction_validated


def test_joint_evidence_units_cannot_be_relabelled_by_input_configuration():
    data = mixed_joint_input()
    with pytest.raises(ValueError, match='coordinate units'):
        evaluate(replace(data, joint_coordinate_units=('rad',) * 17))
    bad = replace(data.residuals['joint'], unit='rad', coordinate_units=None)
    with pytest.raises(ValueError, match='unit'):
        evaluate(replace(data, residuals={'joint': bad}))


@pytest.mark.parametrize('units', [('rad',) * 16, ('m',) * 3 + ('degree',) * 14,
                                    ('stage_distance',) * 3 + ('rad',) * 14])
def test_joint_layout_rejects_unmatched_and_non_si_coordinate_units(units):
    data = mixed_joint_input()
    with pytest.raises(ValueError, match='coordinate units'):
        replace(data, joint_coordinate_units=units)


@pytest.mark.parametrize('references', [(0.,) * 17, (-.1,) * 17, (math.inf,) * 17,
                                       (.1,) * 16, (True,) * 17])
def test_joint_references_are_positive_finite_same_axis_experimental_configuration(references):
    data = mixed_joint_input()
    with pytest.raises(ValueError, match='joint_reference'):
        replace(data, joint_reference_per_dof=references)


def test_joint_reference_source_is_required_and_inputs_are_detached():
    data = mixed_joint_input()
    with pytest.raises(ValueError, match='joint_reference_source'):
        replace(data, joint_reference_source='')
    reference = list(data.joint_reference_per_dof)
    detached = replace(data, joint_reference_per_dof=reference)
    reference[0] = 123.
    assert detached.joint_reference_per_dof[0] == .01
