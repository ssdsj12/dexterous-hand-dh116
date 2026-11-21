"""Synthetic CPU endpoints test reward software; never native Hold/source proof."""
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import hashlib
import json
import math
import subprocess

import pytest

try:
    from go2_pvcnn.control.dh116 import tabletop_training_reward as rw
except ImportError:
    rw = None

ROOT = Path(__file__).resolve().parents[1]
SMOKE = "approach_wiring_smoke_v1"
FULL = "flexible_train_r1_full"


def config(profile=SMOKE):
    return rw.load_config(ROOT / "configs/dh116_training" / (profile + ".json"))


def stamp(tick=0, episode="cpu-episode"):
    return rw.SourceStamp(episode, tick, tick / 480, "cpu_fixture", "cpu_fixture_reader")


def measured(raw, tick=0, unit="dimensionless", *, episode="cpu-episode", applicable=True, reason=None):
    return rw.Measurement(raw, unit, stamp(tick, episode), valid=True,
                          applicable=applicable, history_reason=reason)


def anchors(tick, *, drift=0., uncertainty=0., uid="original", reference=(0., 0., 0.)):
    return tuple(rw.SlipAnchor(side, kind, uid + side + kind, reference, 0,
                  measured(drift, tick, "m"), measured(uncertainty, tick, "m"))
                 for side in ("left", "right") for kind in ("material", "tangential"))


def endpoint(tick=0, *, full=False, distance=.01, contact=False, lifted=False,
             slip=0., failure=False, prohibited=False, health=True, hold_motion=True,
             deadline=False, with_anchors=False, **overrides):
    fields = {"distance": measured(distance, tick, "m"),
              "numerical_health": measured(health, tick),
              "constraint_failure": measured(failure, tick),
              }
    if full:
        fields.update({"prohibited_support": measured(prohibited, tick),
                       "median_z": measured(.04 if lifted else 0., tick, "m"),
                       "min_z": measured(.03 if lifted else 0., tick, "m"),
                       "coverage_left": measured(float(contact), tick),
                       "coverage_right": measured(float(contact), tick),
                       "opposed_left": measured(contact, tick),
                       "opposed_right": measured(contact, tick),
                       "bilateral_support": measured(contact, tick),
                       "off_table": measured(lifted, tick),
                       "no_external_support": measured(not prohibited, tick),
                       "hold_motion": measured(hold_motion, tick),
                       "slip": measured(None, tick, "m", applicable=False,
                                        reason="no_bilateral_contact_since_reset")})
    fields.update(overrides)
    return rw.RewardEndpoint(stamp(tick), ("fixed-material-1", "fixed-material-2"), fields,
                             anchors(tick, drift=slip) if contact or with_anchors else (),
                             (("left", ("thumb", "index")), ("right", ("thumb", "index"))) if full else ())


def request(token, tick, values=None, policy="cpu-policy"):
    return rw.SubmittedRequest(token, policy, stamp(tick), tuple([0.] * 24 if values is None else values),
                               "normalized_submitted_command24_preprojection")


def engine(profile=SMOKE):
    value = rw.TrainingReward(config(profile))
    value.reset_from_verified_state(endpoint(full=profile == FULL), previous_request=request("reset", 0))
    return value


def window(value, token="control-0", *, count=8, start=0, values=None, **kwargs):
    value.begin(request(token, start, values), remaining_substeps=8)
    for tick in range(start + 1, start + count + 1):
        value.observe(endpoint(tick, full=value.config.profile == FULL, **kwargs))
    return value.settle_once(token)


def test_smoke_distance_progress_uses_same_ppo_gamma():
    assert rw is not None, "C03 reward engine missing"
    assert rw.control_total(rw.smoke_potential(.01, .01), rw.smoke_potential(.005, .01),
                            [.10] * 8, .995, 0., False) == pytest.approx(.995 - .1 / 60)


def test_true_failure_integrates_only_three_actual_substeps():
    assert rw.control_total(1., 2., [.15] * 3, .995, -40., True) == pytest.approx(-41. - .15 * 3 / 480)


def test_potential_contact_loop_has_no_discounted_profit():
    gamma = .999
    gain = rw.control_total(0., 2., [0.] * 8, gamma, 0., False)
    loss = rw.control_total(2., 0., [0.] * 8, gamma, 0., False)
    assert gain + gamma * loss == pytest.approx(0.)


@pytest.mark.parametrize("values", [[0.] * 23, [math.nan] * 24, [1.01] * 24])
def test_command_requires_original_finite_bounded_24(values):
    with pytest.raises(ValueError):
        rw.command_rate(values, [0.] * 24)


def test_command_rate_normalizes_difference_by_two():
    assert rw.command_rate([1.] * 24, [-1.] * 24) == 1.


def test_configs_pin_design_schema_thresholds_and_old_flags():
    for profile, gamma in ((SMOKE, .995), (FULL, .999)):
        c = config(profile)
        data = json.loads((ROOT / "configs/dh116_training" / (profile + ".json")).read_text())
        assert c.gamma == gamma == c.ppo["gamma"]
        assert c.config_sha256 == hashlib.sha256((ROOT / "configs/dh116_training" / (profile + ".json")).read_bytes()).hexdigest()
        assert data["schema_sha256"] == rw.SCHEMA_SHA256
        assert data["design_document_sha256"] == "a6a04d7dab20048d22b152762eb07018c3269cd63bfad77ed2b11799d7ed97d9"
        assert data["weight_status"] == "design_approved_native_unverified"
        assert data["legacy_r1_enabled"] is data["legacy_r2_enabled"] is False
        assert data["success"] == {"net_lift_m": .03, "hold_s": 5., "max_slip_m": .0005}
    assert config().ppo["rollout_steps"] == 32
    assert config().ppo["kl_backtrack_steps"] == 3


def test_loader_refuses_gamma_cost_bound_or_second_gamma(tmp_path):
    data = json.loads((ROOT / "configs/dh116_training" / (FULL + ".json")).read_text())
    data["ppo"]["gamma"] = .9995
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="bound"):
        rw.load_config(path)
    data["ppo"]["gamma"] = .999
    data["reward_gamma"] = .999
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="gamma"):
        rw.load_config(path)
    with pytest.raises(ValueError, match="gamma"):
        rw.load_config(ROOT / "configs/dh116_training" / (FULL + ".json"), ppo_gamma=.995)


def test_loader_rejects_changed_original_success_threshold(tmp_path):
    data = json.loads((ROOT / "configs/dh116_training" / (FULL + ".json")).read_text())
    data["success"]["max_slip_m"] = .001
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="success"):
        rw.load_config(path)


def test_smoke_packet_is_owned_provenanced_and_not_native_admission():
    value = engine()
    packet = window(value, distance=.005)
    assert packet.reward == pytest.approx(.995 - .1 / 60)
    assert packet.duration_s == pytest.approx(8 / 480)
    assert packet.phi_pre == 0 and packet.phi_post_raw == 1 and packet.endpoint_phi == 1
    assert packet.token == "control-0" and packet.policy_uid == "cpu-policy"
    assert packet.config_sha256 == value.config.config_sha256
    assert packet.post_context["distance_ref_m"] == .01
    assert packet.post_context["previous_request"] == (0.,) * 24
    assert not packet.native_ready and not packet.teacher_admission and not packet.native_qualification
    assert packet.scope == "cpu_fixture" and not value.native_ready
    assert packet.terms["hold"]["source"] == "profile_excluded"
    assert all(set(("raw", "unit", "valid", "applicable", "source", "contribution")) <= set(term)
               for term in packet.terms.values())
    with pytest.raises(TypeError):
        packet.post_context["distance_ref_m"] = .005
    with pytest.raises(FrozenInstanceError):
        packet.reward = 9


def test_request_tokens_exactly_once_and_projection_targets_rejected():
    value = engine()
    window(value)
    with pytest.raises(RuntimeError, match="token|settled"):
        value.settle_once("control-0")
    with pytest.raises(RuntimeError, match="token"):
        value.begin(request("control-0", 8), remaining_substeps=8)
    with pytest.raises(rw.SourceFailure, match="preprojection"):
        value.begin(replace(request("next", 8), representation="sdk_targets"), remaining_substeps=8)


def test_command_cost_uses_submitted_request_and_previous_owner_request():
    value = engine()
    first = window(value, values=[1.] * 24)
    assert first.cost_per_second == pytest.approx((.1125,) * 8)
    second = window(value, "next", start=8, values=[-1.] * 24)
    assert second.cost_per_second == pytest.approx((.15,) * 8)


def test_external_budget_only_stops_before_a_complete_window():
    value = engine()
    with pytest.raises(ValueError, match="eight|8"):
        value.begin(request("too-short", 0), remaining_substeps=7)
    value.begin(request("control", 0), remaining_substeps=8)
    for tick in range(1, 4):
        value.observe(endpoint(tick))
    with pytest.raises(RuntimeError, match="eight|8|incomplete"):
        value.settle_once("control", truncated=True)


def test_terminal_three_of_eight_retains_raw_phi_and_zeroes_bootstrap():
    value = engine()
    value.begin(request("hard-failure", 0), remaining_substeps=8)
    for tick in range(1, 4):
        value.observe(endpoint(tick, distance=.005, failure=tick == 3))
    packet = value.settle_once("hard-failure")
    assert packet.reward == pytest.approx(-40. - .1 * 3 / 480)
    assert packet.phi_post_raw == 1. and packet.endpoint_phi == 0
    assert packet.terminated and packet.done and not packet.truncated
    assert packet.bootstrap(123.) == 0.
    assert packet.duration_s == pytest.approx(3 / 480)
    with pytest.raises(RuntimeError):
        value.begin(request("after-terminal", 3), remaining_substeps=8)


@pytest.mark.parametrize("truncated,buffer_full,done", [(True, False, True), (False, True, False)])
def test_truncation_buffer_keep_real_phi_and_shaped_value_once(truncated, buffer_full, done):
    value = engine()
    value.begin(request("boundary", 0), remaining_substeps=8)
    for tick in range(1, 9):
        value.observe(endpoint(tick, distance=.005))
    packet = value.settle_once("boundary", truncated=truncated, buffer_full=buffer_full)
    assert packet.endpoint_phi == 1. and packet.done is done
    assert packet.bootstrap(3.) == pytest.approx(.995 * 3.)
    assert packet.reward == pytest.approx(.995 - .1 / 60)


@pytest.mark.parametrize("field", ["distance", "constraint_failure", "numerical_health"])
def test_source_corruption_quarantines_before_event_and_emits_no_packet(field):
    value = engine()
    value.begin(request("bad-source", 0), remaining_substeps=8)
    bad = replace(measured(0., 1), raw=math.nan)
    if field == "distance": bad = replace(bad, unit="m")
    row = endpoint(1, failure=True, **{field: bad})
    with pytest.raises(rw.SourceFailure): value.observe(row)
    assert value.quarantined and value.failed_endpoint is row
    with pytest.raises(rw.SourceFailure): value.settle_once("bad-source")


@pytest.mark.parametrize("bad", ["clock", "target", "unit", "source", "scope"])
def test_same_clock_fixed_target_and_measurement_identity_are_enforced(bad):
    value = engine()
    value.begin(request("bad-identity", 0), remaining_substeps=8)
    row = endpoint(1)
    if bad == "clock": row = endpoint(2)
    if bad == "target": row = replace(row, target_ids=("easiest-new-node",))
    fields = dict(row.fields)
    if bad == "unit": fields["distance"] = replace(fields["distance"], unit="cm")
    if bad == "source": fields["distance"] = replace(fields["distance"], stamp=replace(stamp(1), source=""))
    if bad == "scope": fields["distance"] = replace(fields["distance"], stamp=replace(stamp(1), scope="native_qualified"))
    row = replace(row, fields=fields)
    with pytest.raises(rw.SourceFailure): value.observe(row)


def test_reset_distance_scale_is_fixed_and_floor_is_five_mm():
    value = engine()
    window(value, distance=.02)
    packet = window(value, "progress", start=8, distance=.005)
    assert packet.phi_pre == 0. and packet.phi_post_raw == 1.
    assert packet.post_context["distance_ref_m"] == .01
    other = rw.TrainingReward(config())
    other.reset_from_verified_state(endpoint(distance=.001), previous_request=request("reset", 0))
    assert window(other, distance=0).post_context["distance_ref_m"] == .005


def test_full_known_negative_and_proven_precontact_na_are_valid():
    packet = window(engine(FULL))
    assert packet.reward == pytest.approx(-.1 / 60)
    assert packet.terms["slip"]["raw"] is None and not packet.terms["slip"]["applicable"]
    assert packet.terms["contact"]["raw"] == 0.
    assert packet.terms["lift"]["raw"] == 0.


def test_full_applicable_unknown_is_not_negative_reward_or_usable_packet():
    value = engine(FULL)
    value.begin(request("unknown", 0), remaining_substeps=8)
    with pytest.raises(rw.SourceFailure):
        value.observe(endpoint(1, full=True, coverage_left=measured(None, 1)))
    assert value.quarantined


def test_full_contact_persistence_first_endpoint_is_zero_and_declines():
    value = engine(FULL)
    packet = window(value, contact=True)
    assert packet.post_context["contact_s"] == pytest.approx(7 / 480)
    assert packet.phi_post_raw == pytest.approx(3. + .5 * (7 / 480) / .25)
    lost = window(value, "lost", start=8, with_anchors=True)
    assert lost.phi_post_raw == 0. and lost.reward < -3.
    assert lost.post_context["contact_s"] == 0.


def test_full_lift_uses_current_min_and_median_never_history_peak():
    value = engine(FULL)
    high = window(value, contact=True, lifted=True)
    value.begin(request("lower", 8), remaining_substeps=8)
    for tick in range(9, 17):
        value.observe(endpoint(tick, full=True, contact=True, lifted=True,
                               median_z=measured(.01, tick, "m")))
    lower = value.settle_once("lower")
    assert lower.terms["lift"]["raw"] == pytest.approx(.01)
    assert lower.terms["lift"]["contribution"] == pytest.approx(4. / 3.)
    assert lower.phi_post_raw < high.phi_post_raw and lower.reward < 0.


def test_cpu_full_hold_requires_2400_intervals_and_2401_endpoints():
    value = engine(FULL)
    packet = None
    for start in range(0, 2400, 8):
        packet = window(value, str(start), start=start, contact=True, lifted=True)
    assert packet.post_context["hold_s"] == pytest.approx(2399 / 480)
    assert packet.event == 0. and not packet.terminated
    value.begin(request("success", 2400), remaining_substeps=8)
    value.observe(endpoint(2401, full=True, contact=True, lifted=True))
    result = value.settle_once("success")
    assert result.post_context["hold_s"] == 5.
    assert result.event == 100. and result.terminated and result.endpoint_phi == 0.
    assert result.phi_post_raw == pytest.approx(11.5)
    assert result.scope == "cpu_fixture" and not result.native_qualification


def test_hold_interruption_is_negative_progress_and_restarts_from_zero():
    value = engine(FULL)
    first = window(value, contact=True, lifted=True)
    lost = window(value, "interrupt", start=8, contact=True, lifted=True, hold_motion=False)
    assert lost.post_context["hold_s"] == 0 and lost.phi_post_raw < first.phi_post_raw + .5
    value.begin(request("restart", 16), remaining_substeps=8)
    value.observe(endpoint(17, full=True, contact=True, lifted=True))
    assert value.post_context["hold_s"] == 0.


def test_persistent_material_anchors_survive_loss_reacquisition_and_no_reanchor_cheat():
    value = engine(FULL)
    window(value, contact=True, slip=.0002)
    window(value, "lost", start=8, with_anchors=True, slip=.0003)
    again = window(value, "again", start=16, contact=True, slip=.0006)
    assert again.terms["slip"]["raw"] == pytest.approx(.0006)
    assert not again.terminated  # slipping is not a newly invented terminal guard
    value.begin(request("cheat", 24), remaining_substeps=8)
    row = replace(endpoint(25, full=True, contact=True), anchors=anchors(25, reference=(1., 0., 0.)))
    with pytest.raises(rw.SourceFailure, match="anchor|reference"):
        value.observe(row)


def test_missing_latched_anchor_cannot_become_not_applicable():
    value = engine(FULL)
    window(value, contact=True)
    value.begin(request("forget", 8), remaining_substeps=8)
    with pytest.raises(rw.SourceFailure, match="anchor|applicab"):
        value.observe(endpoint(9, full=True))


def test_slip_conservative_bound_and_original_three_value_status():
    assert rw.slip_status(((.0004, .00005),), .0005) is True
    assert rw.slip_status(((.0006, .00005),), .0005) is False
    assert rw.slip_status(((.0005, .00005),), .0005) is None
    value = engine(FULL)
    value.begin(request("slip-bound", 0), remaining_substeps=8)
    for tick in range(1, 9):
        value.observe(replace(endpoint(tick, full=True, contact=True),
                              anchors=anchors(tick, drift=.0004, uncertainty=.0002)))
    packet = value.settle_once("slip-bound")
    assert packet.terms["slip"]["raw"] == pytest.approx(.0006)
    assert packet.cost_per_second == pytest.approx((1.3,) * 8)
    assert packet.post_context["slip_status"] is None


def test_failure_priority_uses_largest_single_penalty_and_retains_all_reasons():
    value = engine(FULL)
    value.begin(request("failures", 0), remaining_substeps=8)
    value.observe(endpoint(1, full=True, failure=True, prohibited=True, health=False))
    result = value.settle_once("failures")
    assert result.event == -50.
    assert set(result.event_reasons) == {"constraint_failure", "prohibited_support", "numerical_health"}
    assert result.terminated


def test_source_corruption_precedes_success_failure_conflict():
    value = engine(FULL)
    value.begin(request("conflict", 0), remaining_substeps=8)
    with pytest.raises(rw.SourceFailure):
        value.observe(endpoint(1, full=True, contact=True, lifted=True, failure=True,
                               hold_motion=replace(measured(True, 1), valid=False)))
    assert value.quarantined


def test_first_profiles_disable_finite_task_deadline_and_keep_explicit_na():
    for profile in (SMOKE, FULL):
        result = window(engine(profile))
        assert result.post_context["finite_task_remaining_s"] is None
        assert result.terms["task_deadline"]["source"] == "profile_excluded"
        assert not result.terminated


def test_existing_r0_null_config_and_sources_are_unchanged():
    result = subprocess.run(["git", "-c", "safe.directory=*", "diff", "3011160cc9df31f9430b167dfd65047845e3e4f5", "--",
                             "configs/dh116_training/privileged_teacher_reward_r0.json",
                             "src/go2_pvcnn/control/dh116/tabletop_deformable_grasp_task.py"],
                            cwd=ROOT, check=True, text=True, capture_output=True)
    assert result.stdout == ""


def test_original_guard_bundle_cannot_disappear_and_existing_support_guard_is_respected():
    value = rw.TrainingReward(config())
    reset = endpoint(original_guard_support=measured(False))
    value.reset_from_verified_state(reset, previous_request=request("reset", 0))
    value.begin(request("missing-guard", 0), remaining_substeps=8)
    with pytest.raises(rw.SourceFailure, match="guard"):
        value.observe(endpoint(1))
    another = rw.TrainingReward(config())
    another.reset_from_verified_state(reset, previous_request=request("reset", 0))
    another.begin(request("existing-support", 0), remaining_substeps=8)
    another.observe(endpoint(1, original_guard_support=measured(True, 1)))
    packet = another.settle_once("existing-support")
    assert packet.event == -40. and packet.terminated
    assert packet.event_reasons == ("original_guard_support",)


def test_known_raw_slip_without_anchors_cannot_be_silently_reclassified_as_na():
    value = engine(FULL)
    value.begin(request("false-na", 0), remaining_substeps=8)
    with pytest.raises(rw.SourceFailure, match="applicab|anchor|NA"):
        row = endpoint(1, full=True)
        value.observe(replace(row, fields={**row.fields, "slip": measured(0., 1, "m")}))


def test_anchor_measurements_can_have_distinct_stable_reader_sources():
    value = engine(FULL)
    value.begin(request("separate-readers", 0), remaining_substeps=8)
    for tick in range(1, 9):
        row = endpoint(tick, full=True, contact=True)
        owned = tuple(replace(a, drift=replace(a.drift, stamp=replace(a.drift.stamp, source="cpu:" + a.side + a.kind)),
                              uncertainty=replace(a.uncertainty, stamp=replace(a.uncertainty.stamp, source="cpu:" + a.side + a.kind)))
                      for a in row.anchors)
        value.observe(replace(row, anchors=owned))
    assert value.settle_once("separate-readers").event == 0


def test_previous_reset_request_token_cannot_be_reused_as_a_new_submission():
    value = engine()
    with pytest.raises(RuntimeError, match="token"):
        value.begin(request("reset", 0), remaining_substeps=8)


def test_packets_preserve_pre_endpoint_context_and_all_raw_guard_evidence():
    value = engine()
    packet = window(value, values=[1.] * 24, distance=.005)
    assert packet.pre_endpoint.stamp.physics_tick == 0
    assert packet.pre_context["previous_request_token"] == "reset"
    assert packet.post_context["previous_request_token"] == "control-0"
    assert packet.pre_context["previous_request"] == (0.,) * 24
    assert packet.post_context["previous_request"] == (1.,) * 24
    assert packet.pre_terms["distance"]["raw"] == .01
    assert packet.terms["raw:numerical_health"]["raw"] is True
    assert packet.terms["raw:constraint_failure"]["raw"] is False
    assert packet.event_terms["success"]["raw"] is None
    assert packet.event_terms["success"]["source"] == "profile_excluded"


def test_post_context_is_copy_on_write_and_hold_interrupt_decreases_reward():
    value = engine(FULL)
    early = window(value, contact=True, lifted=True)
    saved = dict(early.post_context)
    for start in range(8, 128, 8):
        window(value, str(start), start=start, contact=True, lifted=True)
    lost = window(value, "lost-hold", start=128, contact=True, lifted=True, hold_motion=False)
    assert lost.reward < 0 and lost.post_context["hold_s"] == 0
    assert dict(early.post_context) == saved


def test_failure_overrides_success_on_the_qualifying_hold_endpoint():
    value = engine(FULL)
    for start in range(0, 2400, 8):
        window(value, str(start), start=start, contact=True, lifted=True)
    value.begin(request("failure-success", 2400), remaining_substeps=8)
    value.observe(endpoint(2401, full=True, contact=True, lifted=True, failure=True, prohibited=True))
    packet = value.settle_once("failure-success")
    assert packet.event == -50 and "success" not in packet.event_reasons


def test_reset_cannot_reanchor_the_same_episode_and_target_denominators_are_fixed():
    value = engine(FULL)
    window(value, contact=True)
    with pytest.raises(RuntimeError, match="episode|reset"):
        value.reset_from_verified_state(endpoint(8, full=True), previous_request=request("new-reset", 8))
    value.begin(request("changed-pads", 8), remaining_substeps=8)
    with pytest.raises(rw.SourceFailure, match="pad|denominator"):
        value.observe(replace(endpoint(9, full=True, contact=True),
                              required_pad_ids=(("left", ("thumb",)), ("right", ("thumb",)))))


def test_direct_config_construction_cannot_bypass_discount_bound():
    loaded = config(FULL)
    forged = replace(loaded, ppo={**loaded.ppo, "gamma": .9995})
    with pytest.raises(ValueError, match="bound"):
        rw.TrainingReward(forged)


def test_success_constants_are_immutable():
    with pytest.raises(TypeError):
        rw.SUCCESS["hold_s"] = 1.


@pytest.mark.parametrize("stage", ["reset", "begin", "anchor"])
def test_malformed_owned_schema_is_source_failure_and_quarantined(stage):
    value = engine(FULL)
    with pytest.raises(rw.SourceFailure):
        if stage == "reset":
            value.reset_from_verified_state(object(), previous_request=request("bad", 0))
        elif stage == "begin":
            value.begin(object(), remaining_substeps=8)
        else:
            value.begin(request("bad-anchor", 0), remaining_substeps=8)
            value.observe(replace(endpoint(1, full=True, contact=True), anchors=(object(),)))
    assert value.quarantined


def test_all_previous_training_configs_and_original_task_contract_remain_byte_identical():
    result = subprocess.run(["git", "-c", "safe.directory=*", "diff", "--name-only",
                             "--no-renames", "--diff-filter=a", "3011160cc9df31f9430b167dfd65047845e3e4f5", "--",
                             "configs/dh116_training", "configs/dh116_tasks",
                             "src/go2_pvcnn/control/dh116/tabletop_deformable_grasp_task.py"],
                            cwd=ROOT, check=True, text=True, capture_output=True)
    assert result.stdout == ""


def test_numeric_config_mapping_cannot_change_after_constructor():
    loaded = config(FULL)
    with pytest.raises(TypeError):
        loaded.ppo["gamma"] = .9995
    assert loaded.cost_max_per_second == pytest.approx(2.15)
    assert loaded.discounted_cost_bound == pytest.approx(35.8333333333333)
    assert config().discounted_cost_bound == pytest.approx(.5)


@pytest.mark.parametrize("bad_uid", [["before"], {"uid": "before"}, 17, True, None, "", "   "],
                         ids=["list", "dict", "integer", "bool", "null", "empty", "whitespace"])
def test_spec_fix_anchor_reference_uid_requires_nonempty_immutable_string(bad_uid):
    value = engine(FULL)
    value.begin(request("bad-reference-uid", 0), remaining_substeps=8)
    row = endpoint(1, full=True, contact=True)
    bad = replace(row, anchors=(replace(row.anchors[0], reference_uid=bad_uid), *row.anchors[1:]))
    with pytest.raises(rw.SourceFailure, match="reference identity"):
        value.observe(bad)
    assert value.quarantined and value.failed_endpoint is bad
    with pytest.raises(rw.SourceFailure):
        value.settle_once("bad-reference-uid")


class _MutableIdentity:
    """A corrupt source object whose comparisons impersonate a valid identity."""
    def __init__(self, text):
        self.text = text

    def __eq__(self, other):
        return self.text == other


@pytest.mark.parametrize("field,text", [("episode_uid", "cpu-episode"), ("reset_uid", "cpu-reset"),
                                        ("rod_uid", "cpu-rod"), ("source", "cpu_fixture_reader"),
                                        ("scope", "cpu_fixture"), ("frame", "world")])
def test_spec_fix_all_source_stamp_identity_fields_reject_mutable_comparison_objects(field, text):
    value = engine(FULL)
    value.begin(request("mutable-clock-identity", 0), remaining_substeps=8)
    row = endpoint(1, full=True, contact=True)
    row = replace(row, stamp=replace(row.stamp, **{field: _MutableIdentity(text)}))
    with pytest.raises(rw.SourceFailure):
        value.observe(row)
    assert value.quarantined and value.failed_endpoint is row


@pytest.mark.parametrize("field", ["token", "policy_uid"])
def test_spec_fix_request_identities_reject_mutable_objects(field):
    value = engine(FULL)
    bad = replace(request("bad-request-identity", 0), **{field: ["before"]})
    with pytest.raises(rw.SourceFailure):
        value.begin(bad, remaining_substeps=8)
    assert value.quarantined


@pytest.mark.parametrize("change_tick", [1, 5, 8])
def test_spec_fix_proven_na_source_cannot_change_after_reset_or_in_window(change_tick):
    value = engine(FULL)
    value.begin(request("changed-na-source", 0), remaining_substeps=8)
    for tick in range(1, change_tick):
        value.observe(endpoint(tick, full=True))
    row = endpoint(change_tick, full=True)
    changed = replace(row.fields["slip"], stamp=replace(row.fields["slip"].stamp, source="changed-na-reader-" + str(change_tick)))
    row = replace(row, fields={**row.fields, "slip": changed})
    with pytest.raises(rw.SourceFailure, match="episode source changed: slip"):
        value.observe(row)
    assert value.quarantined and value.failed_endpoint is row
    with pytest.raises(rw.SourceFailure):
        value.settle_once("changed-na-source")


def test_spec_fix_legal_na_stays_valid_and_smoke_still_excludes_full_sources():
    full = window(engine(FULL))
    assert full.terms["slip"]["raw"] is None
    assert full.terms["slip"]["applicable"] is False
    assert full.reward == pytest.approx(-.1 / 60)
    smoke = engine()
    smoke.begin(request("excluded-slip-reader", 0), remaining_substeps=8)
    for tick in range(1, 9):
        unknown_excluded = measured(None, tick, "m")
        unknown_excluded = replace(unknown_excluded,
            stamp=replace(unknown_excluded.stamp, source="unused-full-reader-" + str(tick)), valid=False)
        row = endpoint(tick)
        smoke.observe(replace(row, fields={**row.fields, "slip": unknown_excluded}))
    packet = smoke.settle_once("excluded-slip-reader")
    assert packet.terms["slip"]["source"] == "profile_excluded"
    assert packet.reward == pytest.approx(-.1 / 60)


def test_spec_fix_issued_packet_does_not_alias_external_endpoint_containers():
    value = engine(FULL)
    submitted = [0.] * 24
    request_row = rw.SubmittedRequest("owned-containers", "cpu-policy", stamp(0), submitted,
                                     "normalized_submitted_command24_preprojection")
    value.begin(request_row, remaining_substeps=8)
    uid_container = ["owned-reference"]
    reference = [0., 0., 0.]
    target_ids = ["fixed-material-1", "fixed-material-2"]
    pad_ids = [["left", ["thumb", "index"]], ["right", ["thumb", "index"]]]
    for tick in range(1, 9):
        row = endpoint(tick, full=True, contact=True)
        owned_anchors = list(row.anchors)
        owned_anchors[0] = replace(owned_anchors[0], reference_uid=uid_container[0], reference=reference)
        external_fields = dict(row.fields)
        row = replace(row, target_ids=target_ids, required_pad_ids=pad_ids,
                      fields=external_fields, anchors=owned_anchors)
        value.observe(row)
    packet = value.settle_once("owned-containers")
    before = (packet.endpoint, dict(packet.post_context))
    submitted[0] = 1.
    uid_container[0] = "changed-uid"
    reference[0] = 99.
    target_ids[0] = "easiest-new-target"
    pad_ids[0][1].clear()
    external_fields.clear()
    owned_anchors.clear()
    assert packet.endpoint == before[0] and dict(packet.post_context) == before[1]
    assert packet.endpoint.anchors[0].reference_uid == "owned-reference"
    assert packet.endpoint.anchors[0].reference == (0., 0., 0.)
    assert packet.post_context["anchors"][0][1][0] == "owned-reference"
    assert packet.post_context["previous_request"] == (0.,) * 24


@pytest.mark.parametrize("field,text", [("unit", "m"), ("history_reason", "no_bilateral_contact_since_reset")])
def test_spec_fix_na_provenance_metadata_also_rejects_mutable_identities(field, text):
    value = engine(FULL)
    value.begin(request("mutable-na-metadata", 0), remaining_substeps=8)
    row = endpoint(1, full=True)
    bad_na = replace(row.fields["slip"], **{field: _MutableIdentity(text)})
    row = replace(row, fields={**row.fields, "slip": bad_na})
    with pytest.raises(rw.SourceFailure):
        value.observe(row)
    assert value.quarantined and value.failed_endpoint is row


def test_spec_fix_request_representation_cannot_use_mutable_comparison_object():
    value = engine()
    bad = replace(request("mutable-representation", 0),
                  representation=_MutableIdentity("normalized_submitted_command24_preprojection"))
    with pytest.raises(rw.SourceFailure):
        value.begin(bad, remaining_substeps=8)
    assert value.quarantined


def test_spec_fix_required_pad_side_is_an_immutable_identity_before_set_lookup():
    value = engine(FULL)
    value.begin(request("mutable-pad-side", 0), remaining_substeps=8)
    row = replace(endpoint(1, full=True, contact=True),
                  required_pad_ids=((['left'], ('thumb', 'index')), ('right', ('thumb', 'index'))))
    with pytest.raises(rw.SourceFailure):
        value.observe(row)
    assert value.quarantined and value.failed_endpoint is row


@pytest.mark.parametrize("bad_token", [_MutableIdentity("settle-identity"), ["settle-identity"],
                                      {"token": "settle-identity"}, 7, True, None, "", "   "],
                         ids=["mutable-comparison", "list", "dict", "integer", "bool", "null", "empty", "whitespace"])
def test_quality_fix_settle_token_must_be_nonempty_native_string(bad_token):
    value = engine()
    value.begin(request("settle-identity", 0), remaining_substeps=8)
    for tick in range(1, 9):
        value.observe(endpoint(tick))
    last = value._last
    with pytest.raises(rw.SourceFailure, match="token"):
        value.settle_once(bad_token)
    assert value.quarantined and value.failed_endpoint is last
    with pytest.raises(rw.SourceFailure):
        value.settle_once("settle-identity")


class _OverflowFloatInt(int):
    def __float__(self):
        raise OverflowError("unrepresentable source scalar")


_BAD_NUMERICS = [10 ** 1000, -(10 ** 1000), _OverflowFloatInt(1)]
_BAD_NUMERIC_IDS = ["huge-positive", "huge-negative", "overflow-conversion"]


@pytest.mark.parametrize("bad", _BAD_NUMERICS, ids=_BAD_NUMERIC_IDS)
@pytest.mark.parametrize("field", ["distance", "median_z", "min_z", "coverage_left", "coverage_right",
                                   "solver_time", "reference", "drift", "uncertainty"])
def test_quality_fix_invalid_numeric_sources_quarantine_without_overflow_escape(field, bad):
    value = engine(FULL)
    value.begin(request("numeric-source", 0), remaining_substeps=8)
    row = endpoint(1, full=True, contact=True)
    if field == "solver_time":
        row = replace(row, stamp=replace(row.stamp, solver_time_s=bad))
    elif field in ("reference", "drift", "uncertainty"):
        first = row.anchors[0]
        changes = {"reference": (bad, 0., 0.)} if field == "reference" else {
            field: replace(getattr(first, field), raw=bad)}
        row = replace(row, anchors=(replace(first, **changes), *row.anchors[1:]))
    else:
        row = replace(row, fields={**row.fields, field: replace(row.fields[field], raw=bad)})
    with pytest.raises(rw.SourceFailure):
        value.observe(row)
    assert value.quarantined and value.failed_endpoint is row
    with pytest.raises(rw.SourceFailure):
        value.observe(endpoint(1, full=True, contact=True))
    with pytest.raises(rw.SourceFailure):
        value.settle_once("numeric-source")


@pytest.mark.parametrize("bad", _BAD_NUMERICS, ids=_BAD_NUMERIC_IDS)
@pytest.mark.parametrize("stage", ["reset-distance", "reset-request", "begin-request"])
def test_quality_fix_invalid_reset_and_command_numeric_inputs_are_source_failures(stage, bad):
    value = rw.TrainingReward(config()) if stage.startswith("reset") else engine()
    row = endpoint(distance=bad) if stage == "reset-distance" else endpoint()
    bad_request = request("numeric-request", 0, [bad] + [0.] * 23)
    with pytest.raises(rw.SourceFailure):
        if stage == "begin-request":
            value.begin(bad_request, remaining_substeps=8)
        else:
            value.reset_from_verified_state(row, previous_request=bad_request if stage == "reset-request" else request("reset", 0))
    assert value.quarantined
    with pytest.raises(rw.SourceFailure):
        value.begin(request("retry", 0), remaining_substeps=8)


@pytest.mark.parametrize("numeric_literal", ["1" + "0" * 1000, "-1" + "0" * 1000, "1e10000", "-1e10000"])
def test_quality_fix_invalid_config_gamma_numeric_syntax_is_value_error(tmp_path, numeric_literal):
    text = (ROOT / "configs/dh116_training" / (FULL + ".json")).read_text()
    text = text.replace('"gamma": 0.999', '"gamma": ' + numeric_literal)
    assert text != (ROOT / "configs/dh116_training" / (FULL + ".json")).read_text()
    path = tmp_path / "invalid-gamma.json"
    path.write_text(text)
    with pytest.raises(ValueError, match="gamma"):
        rw.load_config(path)


@pytest.mark.parametrize("bad", _BAD_NUMERICS, ids=_BAD_NUMERIC_IDS)
def test_quality_fix_invalid_trainer_gamma_is_rejected_without_conversion(bad):
    with pytest.raises(ValueError, match="gamma"):
        rw.load_config(ROOT / "configs/dh116_training" / (FULL + ".json"), ppo_gamma=bad)


@pytest.mark.parametrize("field", ["unused", "slip", "median_z", "coverage_left"])
@pytest.mark.parametrize("bad_raw", [[1.], {"raw": 1.}, ([1.],), _MutableIdentity("raw")],
                         ids=["list", "dict", "nested-list", "mutable-object"])
def test_quality_fix_unconsumed_and_excluded_raw_cannot_alias_mutable_evidence(field, bad_raw):
    value = engine()
    value.begin(request("mutable-extra", 0), remaining_substeps=8)
    row = endpoint(1)
    row = replace(row, fields={**row.fields, field: replace(measured(bad_raw, 1), valid=False)})
    with pytest.raises(rw.SourceFailure):
        value.observe(row)
    assert value.quarantined and value.failed_endpoint is row
    with pytest.raises(rw.SourceFailure):
        value.settle_once("mutable-extra")


@pytest.mark.parametrize("bad", _BAD_NUMERICS, ids=_BAD_NUMERIC_IDS)
def test_quality_fix_unused_numeric_corruption_is_not_retained_in_packet(bad):
    value = engine()
    value.begin(request("numeric-extra", 0), remaining_substeps=8)
    row = endpoint(1, unused=measured(bad, 1))
    with pytest.raises(rw.SourceFailure):
        value.observe(row)
    assert value.quarantined and value.failed_endpoint is row


def test_quality_fix_valid_tuple_raw_remains_owned_without_smoke_source_prerequisites():
    value = engine()
    container = {"unused": (1., (2., 3.), None, True)}
    value.begin(request("tuple-extra", 0), remaining_substeps=8)
    for tick in range(1, 9):
        excluded = replace(measured(container["unused"], tick, "vector"),
                           valid=False, stamp=replace(stamp(tick), source="excluded-reader-" + str(tick)))
        row = endpoint(tick, unused=excluded, slip=replace(measured(None, tick, "m"), valid=False))
        value.observe(row)
    packet = value.settle_once("tuple-extra")
    container["unused"] = ([99.],)
    assert packet.endpoint.fields["unused"].raw == (1., (2., 3.), None, True)
    assert packet.endpoint.fields["unused"].valid is False
    assert packet.endpoint.fields["unused"].stamp.source == "excluded-reader-8"
    assert packet.terms["slip"]["source"] == "profile_excluded"
    assert packet.reward == pytest.approx(-.1 / 60)


def test_quality_fix_programming_errors_are_not_converted_to_source_failures(monkeypatch):
    value = engine()
    value.begin(request("programming-error", 0), remaining_substeps=8)

    def broken_potential(row, values):
        raise RuntimeError("genuine implementation defect")

    monkeypatch.setattr(value, "_potential", broken_potential)
    with pytest.raises(RuntimeError, match="genuine implementation defect"):
        value.observe(endpoint(1))
    assert not value.quarantined


@pytest.mark.parametrize("field", ["unit", "history_reason", "valid", "applicable", "stamp"])
def test_quality_fix_unused_measurement_metadata_cannot_retain_mutable_objects(field):
    value = engine()
    value.begin(request("mutable-extra-metadata", 0), remaining_substeps=8)
    extra = replace(measured((1., 2.), 1), **{field: ["mutable"]})
    row = endpoint(1, unused=extra)
    with pytest.raises(rw.SourceFailure):
        value.observe(row)
    assert value.quarantined and value.failed_endpoint is row


@pytest.mark.parametrize("field", ["source", "physics_tick", "solver_time_s"])
def test_quality_fix_unused_measurement_stamp_cannot_retain_mutable_objects(field):
    value = engine()
    value.begin(request("mutable-extra-clock", 0), remaining_substeps=8)
    extra = replace(measured((1., 2.), 1), stamp=replace(stamp(1), **{field: ["mutable"]}))
    row = endpoint(1, unused=extra)
    with pytest.raises(rw.SourceFailure):
        value.observe(row)
    assert value.quarantined and value.failed_endpoint is row


@pytest.mark.parametrize("field", ["reference_uid", "reference", "drift"])
def test_quality_fix_excluded_anchor_evidence_cannot_alias_mutable_objects(field):
    value = engine()
    value.begin(request("mutable-excluded-anchor", 0), remaining_substeps=8)
    a = anchors(1)[0]
    changes = {"reference_uid": ["mutable"], "reference": ([0.],),
               "drift": replace(a.drift, raw=[0.])}
    row = replace(endpoint(1), anchors=(replace(a, **{field: changes[field]}),))
    with pytest.raises(rw.SourceFailure):
        value.observe(row)
    assert value.quarantined and value.failed_endpoint is row


def test_quality_fix_excluded_pad_evidence_cannot_alias_mutable_objects():
    value = engine()
    value.begin(request("mutable-excluded-pads", 0), remaining_substeps=8)
    row = replace(endpoint(1), required_pad_ids=(("left", (["mutable"],)),))
    with pytest.raises(rw.SourceFailure):
        value.observe(row)
    assert value.quarantined and value.failed_endpoint is row


def test_quality_fix_excluded_owned_evidence_does_not_require_full_source_semantics():
    value = engine()
    value.begin(request("excluded-structure", 0), remaining_substeps=8)
    for tick in range(1, 9):
        extra = replace(measured((1., 2.), 0, "vector"), valid=False,
                        stamp=replace(stamp(0), source="other-optional-reader-" + str(tick)))
        # Optional data is structurally owned, but not a full measurement/source prerequisite.
        row = replace(endpoint(tick, unused=extra), anchors=(anchors(0)[0],),
                      required_pad_ids=(("left", ("optional",)),))
        value.observe(row)
    packet = value.settle_once("excluded-structure")
    assert packet.endpoint.fields["unused"].raw == (1., 2.)
    assert packet.endpoint.fields["unused"].stamp.physics_tick == 0
    assert packet.endpoint.anchors[0].reference_uid == "originalleftmaterial"
    assert packet.terms["slip"]["source"] == "profile_excluded"


@pytest.mark.parametrize("case", ["distance-ratio", "slip-sum", "slip-status", "reward-difference",
                                   "cost-sum", "event-addition"])
def test_spec_final_fix_pure_math_rejects_nonfinite_derived_values(case):
    with pytest.raises(ValueError):
        if case == "distance-ratio":
            rw.smoke_potential(1e308, .005)
        elif case == "slip-sum":
            rw.conservative_slip(((1e308, 1e308),))
        elif case == "slip-status":
            rw.slip_status(((1e308, 1e308),))
        elif case == "reward-difference":
            rw.control_total(-1e308, 1e308, [0.], .995, 0., False)
        elif case == "cost-sum":
            rw.control_total(0., 0., [1e308] * 8, .995, 0., False)
        else:
            rw.control_total(0., 1e308, [0.], .995, 1e308, False)


@pytest.mark.parametrize("case", ["both-height-deltas", "hidden-median-delta", "hidden-min-delta",
                                   "height-ratio", "slip-sum", "slip-ratio", "distance-ratio"])
@pytest.mark.parametrize("contact", [False, True])
def test_spec_final_fix_finite_sources_cannot_emit_nonfinite_derived_evidence(case, contact):
    value = rw.TrainingReward(config(FULL))
    reset = endpoint(full=True)
    if case == "both-height-deltas":
        reset = replace(reset, fields={**reset.fields, "median_z": measured(-1e308, 0, "m"),
                                      "min_z": measured(-1e308, 0, "m")})
    elif case == "hidden-median-delta":
        reset = replace(reset, fields={**reset.fields, "median_z": measured(-1e308, 0, "m")})
    elif case == "hidden-min-delta":
        reset = replace(reset, fields={**reset.fields, "min_z": measured(-1e308, 0, "m")})
    value.reset_from_verified_state(reset, previous_request=request("reset", 0))
    value.begin(request("derived-overflow", 0), remaining_substeps=8)
    row = endpoint(1, full=True, contact=contact, lifted=contact,
                   with_anchors=case.startswith("slip"))
    if case in ("both-height-deltas", "height-ratio"):
        row = replace(row, fields={**row.fields, "median_z": measured(1e308, 1, "m"),
                                  "min_z": measured(1e308, 1, "m")})
    elif case in ("hidden-median-delta", "hidden-min-delta"):
        field = "median_z" if case == "hidden-median-delta" else "min_z"
        row = replace(row, fields={**row.fields, field: measured(1e308, 1, "m")})
    elif case.startswith("slip"):
        row = replace(row, anchors=anchors(1, drift=1e308,
                                           uncertainty=1e308 if case == "slip-sum" else 0.))
    else:
        row = replace(row, fields={**row.fields, "distance": measured(1e308, 1, "m")})
    if case == "height-ratio" and not contact:
        # Without genuine lift the normalized lift ratio is not evaluated.
        value.observe(row)
        assert math.isfinite(value._terms["lift"]["raw"])
        return
    with pytest.raises(rw.SourceFailure, match="finite|overflow"):
        value.observe(row)
    assert value.quarantined and value.failed_endpoint is row
    with pytest.raises(rw.SourceFailure):
        value.observe(endpoint(1, full=True))
    with pytest.raises(rw.SourceFailure):
        value.settle_once("derived-overflow")


def test_spec_final_fix_finite_saturation_still_uses_approved_clips_and_no_new_slip_terminal():
    assert rw.smoke_potential(1e6, .005) == 0.
    assert rw.conservative_slip(((.01, .001),)) == pytest.approx(.011)
    packet = window(engine(FULL), contact=True, slip=.01)
    assert packet.terms["slip"]["contribution"] == 2.
    assert packet.event == 0. and not packet.terminated


def _mutable_dataclass_subclass(value):
    cls = type("Mutable" + type(value).__name__, (type(value),), {"__setattr__": object.__setattr__})
    from dataclasses import fields
    return cls(**{f.name: getattr(value, f.name) for f in fields(value) if f.init})


@pytest.mark.parametrize("stage", ["reset-endpoint", "reset-stamp", "previous-request", "previous-stamp",
                                    "observe-endpoint", "observe-stamp", "measurement", "measurement-stamp",
                                    "extra-measurement", "extra-stamp", "anchor", "anchor-measurement",
                                    "anchor-stamp", "excluded-anchor", "request", "request-stamp"])
def test_spec_final_fix_mutable_dataclass_subclasses_are_rejected_at_ingress(stage):
    value = rw.TrainingReward(config(FULL)) if stage.startswith(("reset", "previous")) else engine(FULL)
    row = endpoint(full=True) if stage.startswith(("reset", "previous")) else endpoint(1, full=True, contact=True)
    submitted = request("reset" if stage.startswith(("reset", "previous")) else "subclass", 0)
    with pytest.raises(rw.SourceFailure):
        if stage.startswith(("reset", "previous")):
            if stage == "reset-endpoint":
                row = _mutable_dataclass_subclass(row)
            elif stage == "reset-stamp":
                row = replace(row, stamp=_mutable_dataclass_subclass(row.stamp))
            elif stage == "previous-request":
                submitted = _mutable_dataclass_subclass(submitted)
            else:
                submitted = replace(submitted, stamp=_mutable_dataclass_subclass(submitted.stamp))
            value.reset_from_verified_state(row, previous_request=submitted)
        elif stage.startswith("request"):
            submitted = (_mutable_dataclass_subclass(submitted) if stage == "request" else
                         replace(submitted, stamp=_mutable_dataclass_subclass(submitted.stamp)))
            value.begin(submitted, remaining_substeps=8)
        else:
            if stage == "excluded-anchor":
                value = engine()
            value.begin(submitted, remaining_substeps=8)
            if stage == "observe-endpoint":
                row = _mutable_dataclass_subclass(row)
            elif stage == "observe-stamp":
                row = replace(row, stamp=_mutable_dataclass_subclass(row.stamp))
            elif stage in ("measurement", "measurement-stamp", "extra-measurement", "extra-stamp"):
                m = row.fields["distance"]
                m = (_mutable_dataclass_subclass(m) if stage.endswith("measurement") else
                     replace(m, stamp=_mutable_dataclass_subclass(m.stamp)))
                name = "unused" if stage.startswith("extra") else "distance"
                row = replace(row, fields={**row.fields, name: m})
            else:
                a = row.anchors[0]
                if stage in ("anchor", "excluded-anchor"):
                    a = _mutable_dataclass_subclass(a)
                elif stage == "anchor-measurement":
                    a = replace(a, drift=_mutable_dataclass_subclass(a.drift))
                else:
                    a = replace(a, drift=replace(a.drift, stamp=_mutable_dataclass_subclass(a.drift.stamp)))
                row = replace(endpoint(1), anchors=(a,)) if stage == "excluded-anchor" else replace(
                    row, anchors=(a, *row.anchors[1:]))
            value.observe(row)
    assert value.quarantined
    with pytest.raises(rw.SourceFailure):
        value.settle_once(submitted.token)


def test_spec_final_fix_mutable_config_subclass_cannot_change_issued_provenance():
    with pytest.raises(TypeError):
        rw.TrainingReward(_mutable_dataclass_subclass(config()))


@pytest.mark.parametrize("field", ["physics_tick", "env_id"])
def test_spec_final_fix_unrepresentable_clock_numeric_metadata_is_quarantined(field):
    value = rw.TrainingReward(config())
    row = endpoint()
    row = replace(row, stamp=replace(row.stamp, **{field: 10 ** 1000}))
    row = replace(row, fields={name: replace(m, stamp=row.stamp) for name, m in row.fields.items()})
    previous = replace(request("reset", 0), stamp=row.stamp)
    with pytest.raises(rw.SourceFailure):
        value.reset_from_verified_state(row, previous_request=previous)
    assert value.quarantined and value.failed_endpoint is row


def test_spec_final_fix_valid_packet_contains_only_finite_numeric_evidence():
    from collections.abc import Mapping
    from dataclasses import fields, is_dataclass

    packet = window(engine(FULL), contact=True, lifted=True)
    pending = [packet]
    while pending:
        value = pending.pop()
        if is_dataclass(value):
            pending.extend(getattr(value, f.name) for f in fields(value))
        elif isinstance(value, Mapping):
            pending.extend(value.values())
        elif type(value) is tuple:
            pending.extend(value)
        elif type(value) in (int, float):
            assert math.isfinite(value)


def test_spec_final_fix_smoke_excludes_full_derived_math_even_with_large_owned_optional_values():
    value = engine()
    value.begin(request("excluded-full-overflow", 0), remaining_substeps=8)
    for tick in range(1, 9):
        row = endpoint(tick, median_z=measured(1e308, tick, "m"), min_z=measured(-1e308, tick, "m"))
        row = replace(row, anchors=anchors(tick, drift=1e308, uncertainty=1e308))
        value.observe(row)
    packet = value.settle_once("excluded-full-overflow")
    assert packet.reward == pytest.approx(-.1 / 60)
    assert packet.terms["lift"]["source"] == packet.terms["slip"]["source"] == "profile_excluded"
    assert packet.endpoint.anchors[0].drift.raw == 1e308
    assert not value.quarantined


@pytest.mark.parametrize("field,bad", [
    ("config_sha256", ["before"]), ("config_sha256", {"sha": "before"}),
    ("config_sha256", ""), ("config_sha256", "a" * 63), ("config_sha256", "g" * 64),
    ("config_sha256", True), ("config_sha256", _MutableIdentity("a" * 64)),
    ("profile_version", []), ("profile_version", True), ("profile_version", 1.),
    ("profile_version", 2), ("profile", _MutableIdentity(SMOKE)),
    ("schema_sha256", _MutableIdentity(rw.SCHEMA_SHA256)),
    ("schema_sha256", ["schema"]), ("schema_sha256", "a" * 64),
    ("cost_max_per_second", []), ("cost_max_per_second", math.inf),
    ("cost_max_per_second", math.nan), ("cost_max_per_second", 10 ** 1000),
    ("cost_max_per_second", .25), ("cost_max_per_second", True),
    ("discounted_cost_bound", math.inf), ("discounted_cost_bound", math.nan),
    ("discounted_cost_bound", 10 ** 1000), ("discounted_cost_bound", True),
], ids=["sha-list", "sha-dict", "sha-empty", "sha-short", "sha-nonhex", "sha-bool", "sha-object",
        "version-list", "version-bool", "version-float", "version-unsupported", "profile-object",
        "schema-object", "schema-list", "schema-wrong", "maximum-list", "maximum-inf",
        "maximum-nan", "maximum-huge", "maximum-wrong", "maximum-bool",
        "bound-inf", "bound-nan", "bound-huge", "bound-bool"])
def test_config_origin_fix_all_retained_metadata_is_primitive_and_validated(field, bad):
    candidate = replace(config(), **{field: bad})
    with pytest.raises(ValueError):
        rw.TrainingReward(candidate)


@pytest.mark.parametrize("field,changes", [
    ("weights", {"distance": [2.]}), ("weights", {"distance": 200.}),
    ("weights", {"unused": {"mutable": [1.]}}),
    ("costs", {"time": -.1}), ("costs", {"time": {"mutable": [1.]}}),
    ("costs", {"slip": [None]}), ("costs", {"velocity": 0.}),
    ("events", {"trusted_constraint_failure": -400.}), ("events", {"success": 100.}),
    ("events", {"unused": ["mutable"]}),
    ("ppo", {"learning_rate": [1e-4]}), ("ppo", {"learning_rate": 2e-4}),
    ("ppo", {"epochs": True}), ("ppo", {"epochs": 1.}),
    ("ppo", {"unused": {"mutable": [1.]}}),
], ids=["weight-list", "weight-changed", "weight-extra-nested", "cost-negative", "cost-nested",
        "cost-list", "cost-na-changed", "event-changed", "event-na-changed", "event-extra-list",
        "ppo-list", "ppo-changed", "ppo-bool", "ppo-float-count", "ppo-extra-nested"])
def test_config_origin_fix_mapping_schema_cannot_retain_mutable_or_unapproved_values(field, changes):
    loaded = config()
    candidate = replace(loaded, **{field: {**getattr(loaded, field), **changes}})
    with pytest.raises(ValueError):
        rw.TrainingReward(candidate)


@pytest.mark.parametrize("kind", ["replace", "constructor", "plausible-sha", "self-consistent-gamma"])
def test_config_origin_fix_data_construction_does_not_create_file_provenance(kind):
    from dataclasses import fields

    loaded = config()
    if kind == "replace":
        candidate = replace(loaded)
    elif kind == "constructor":
        candidate = rw.RewardConfig(**{f.name: getattr(loaded, f.name) for f in fields(loaded) if f.init})
    elif kind == "plausible-sha":
        candidate = replace(loaded, config_sha256="a" * 64)
    else:
        gamma = .99
        candidate = replace(loaded, ppo={**loaded.ppo, "gamma": gamma},
                            discounted_cost_bound=loaded.cost_max_per_second / (60 * (1 - gamma)))
    with pytest.raises(ValueError, match="source|provenance|loader|file|SHA"):
        rw.TrainingReward(candidate)


@pytest.mark.parametrize("change", ["whitespace", "gamma", "deleted"])
def test_config_origin_fix_engine_rechecks_actual_source_file_bytes(tmp_path, change):
    path = tmp_path / "source-profile.json"
    path.write_bytes((ROOT / "configs/dh116_training" / (SMOKE + ".json")).read_bytes())
    loaded = rw.load_config(path)
    if change == "deleted":
        path.unlink()
    elif change == "whitespace":
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        data = json.loads(path.read_bytes())
        data["ppo"]["gamma"] = .99
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="source|provenance|file|bytes|SHA"):
        rw.TrainingReward(loaded)


@pytest.mark.parametrize("field,subfield,bad", [
    ("profile_version", None, True), ("profile_version", None, 1.),
    ("ppo", "epochs", True), ("ppo", "epochs", 3.),
    ("potential_weights", "geometry", True),
    ("costs_per_second", "slip", True), ("costs_per_second", "velocity", False),
], ids=["version-bool", "version-float", "ppo-bool-count", "ppo-float-count",
        "weight-bool", "cost-bool", "velocity-bool"])
def test_config_origin_fix_actual_loader_rejects_schema_type_aliases(tmp_path, field, subfield, bad):
    data = json.loads((ROOT / "configs/dh116_training" / (FULL + ".json")).read_bytes())
    if subfield is None:
        data[field] = bad
    else:
        data[field][subfield] = bad
    path = tmp_path / "typed-profile.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        rw.load_config(path)


@pytest.mark.parametrize("profile", [SMOKE, FULL])
def test_config_origin_fix_actual_loader_file_and_packet_identity_are_immutable(tmp_path, profile):
    path = tmp_path / (profile + ".json")
    raw = (ROOT / "configs/dh116_training" / (profile + ".json")).read_bytes()
    path.write_bytes(raw)
    loaded = rw.load_config(path)
    expected_sha = hashlib.sha256(raw).hexdigest()
    for name in ("ppo", "weights", "costs", "events"):
        values = getattr(loaded, name)
        assert all(type(k) is str and (v is None or type(v) in (int, float)) for k, v in values.items())
        with pytest.raises(TypeError):
            values["external"] = [1.]
    value = rw.TrainingReward(loaded)
    value.reset_from_verified_state(endpoint(full=profile == FULL), previous_request=request("reset", 0))
    packet = window(value)
    assert type(packet.config_sha256) is str and packet.config_sha256 == expected_sha
    assert type(packet.profile_version) is int and packet.profile_version == 1
    assert packet.post_context["config_sha256"] == expected_sha
    assert packet.schema_sha256 == rw.SCHEMA_SHA256
    assert packet.reward == pytest.approx(-.1 / 60)
    # A separately loaded valid file revision has its own actual bytes identity.
    path.write_bytes(raw + b"\n")
    revised = rw.load_config(path)
    assert revised.config_sha256 != expected_sha
    assert packet.config_sha256 == packet.post_context["config_sha256"] == expected_sha
    assert rw.TrainingReward(revised).config.config_sha256 == hashlib.sha256(raw + b"\n").hexdigest()


@pytest.mark.parametrize("change", ["sha", "self-consistent-gamma"])
def test_config_origin_fix_copied_source_claim_still_requires_actual_bytes(change):
    from copy import copy

    loaded = config()
    candidate = copy(loaded)
    if change == "sha":
        object.__setattr__(candidate, "config_sha256", "a" * 64)
    else:
        gamma = .99
        object.__setattr__(candidate, "ppo", {**candidate.ppo, "gamma": gamma})
        object.__setattr__(candidate, "discounted_cost_bound", candidate.cost_max_per_second / (60 * (1 - gamma)))
    with pytest.raises(ValueError, match="source|bytes|SHA"):
        rw.TrainingReward(candidate)


def test_config_origin_fix_revalidated_copy_cannot_alias_engine_owned_config():
    from copy import copy

    candidate = copy(config())
    value = rw.TrainingReward(candidate)
    expected_sha = candidate.config_sha256
    # Even a caller bypassing its copy's frozen API cannot alter the fresh owner.
    object.__setattr__(candidate, "config_sha256", ["external-mutation"])
    object.__setattr__(candidate, "profile_version", ["external-mutation"])
    value.reset_from_verified_state(endpoint(), previous_request=request("reset", 0))
    packet = window(value)
    assert packet.config_sha256 == packet.post_context["config_sha256"] == expected_sha
    assert type(packet.config_sha256) is str and packet.profile_version == 1


@pytest.mark.parametrize("tick, raw, scope, source, accepted", [
    (18, 0.03333333507180214, "unqualified_source", "policy_owned_pinch_endpoint_v1", True),
    (19, 0.03333333507180214, "unqualified_source", "policy_owned_pinch_endpoint_v1", False),
    (18, 0.03541666641831398, "unqualified_source", "policy_owned_pinch_endpoint_v1", False),
    (18, 0.03333333507180214, "cpu_fixture", "cpu_fixture_reader", False),
    (18, 0.03333333507180214, "unqualified_source", "other_clock_reader", False),
    (18, 0.03333333507180214, "unqualified_source", "policy_owned_tripod_endpoint_v1", True),
    (19, 0.03333333507180214, "unqualified_source", "policy_owned_tripod_endpoint_v1", False),
    (18, 0.03541666641831398, "unqualified_source", "policy_owned_tripod_endpoint_v1", False),
    (18, 0.03333333507180214 + 5e-12, "unqualified_source", "policy_owned_tripod_endpoint_v1", False),
])
def test_native_smoke_float32_clock_only_accepts_consecutive_physics(tick, raw, scope, source, accepted):
    def sample(tick, seconds):
        ep = endpoint(tick)
        clock = replace(ep.stamp, solver_time_s=seconds, scope=scope, source=source)
        return replace(ep, stamp=clock, fields={k: replace(v, stamp=clock) for k, v in ep.fields.items()})

    value = rw.TrainingReward(config())
    start = sample(17, 0.03125)
    value.reset_from_verified_state(start, previous_request=replace(request("reset", 17), stamp=start.stamp))
    value.begin(replace(request("actual-control", 17), stamp=start.stamp), remaining_substeps=8)
    if accepted:
        value.observe(sample(tick, raw))
        assert value._last.stamp.solver_time_s == raw
    else:
        with pytest.raises(rw.SourceFailure, match="consecutive actual 480Hz"):
            value.observe(sample(tick, raw))
