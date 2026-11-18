"""CPU-only event, validity and duration contracts for grasp diagnostics."""
import copy
import importlib
import importlib.util
import json
import math

import pytest

MODULE = "go2_pvcnn.control.dh116.tabletop_grasp_contact_diagnostics"
HANDS = ("left", "right")
DIGITS = ("thumb", "index", "middle", "ring", "little")
REASONS = {"NO_CONTACT", "ONE_SIDE_CONTACT", "WRONG_NORMAL", "CONTACT_LOSS",
           "NON_OPPOSED", "TORQUE_LIMITED", "SLIP"}
TORQUE_SOURCE = "implicit_actuator_estimate_not_hardware_measurement"


def reducer(**kwargs):
    assert importlib.util.find_spec(MODULE) is not None, "missing pure CPU contact diagnostic reducer"
    return importlib.import_module(MODULE).GraspContactDiagnostics("cpu-test", **kwargs)


def row(*, env=0, episode=0, style="pinch_thumb_index", contacts=True):
    pads = []
    for hi, hand in enumerate(HANDS):
        for di, digit in enumerate(DIGITS):
            bearing = contacts and di < 2
            pads.append(dict(hand=hand, digit=digit, body_id=hi * 5 + di,
                valid=True, raw_load_n=1. if bearing else 0., load_n=1. if bearing else 0.,
                accepted_load_n=1. if bearing else 0.,
                position_w_m=[0., 0., 0.] if bearing else None,
                normal_on_hand_w=([1., 0., 0.] if di == 0 else [-1., 0., 0.]) if bearing else None,
                contact_count=int(bearing), wrong_normal_count=0,
                raw_contact_ids=[hi * 5 + di] if bearing else []))
    return dict(env_id=env, episode_id=episode, grasp_style=style,
        live=True, done=False, native_success=False, measurement_valid=True,
        pads=pads, raw_contacts=[], estimated_torque_nm=[0.] * 12,
        effort_limit_nm=[1.] * 12, torque_valid=[True] * 12, torque_source=TORQUE_SOURCE,
        slip_m=None, slip_valid=False, slip_applicable=False, slip_reason="no_reference")


def frame(step=0, *rows, dt=.01):
    return dict(physics_step=step, control_step=step // 2, control_substep=step % 2,
                sim_time_s=(step + 1) * dt, dt_s=dt, rows=list(rows) or [row()])


def selected_events(record, kind):
    return [event for event in record["events"] if event["event_type"] == kind]


def test_missing_api_is_a_test_failure_and_cpu_import_is_sufficient():
    d = reducer()
    assert d.summary()["by_style"]["pinch_thumb_index"]["coverage"] is None
    assert d.summary()["by_style"]["tripod_thumb_index_middle"]["valid_s"] == 0.


def test_contact_establish_persist_loss_have_exact_identity_and_si_duration():
    records, events = [], []
    d = reducer(sink=records.append, event_sink=events.append)
    first = d.observe(frame(0))[0]
    second = d.observe(frame(1))[0]
    gone = d.observe(frame(2, row(contacts=False)))[0]
    assert len(selected_events(first, "CONTACT_ESTABLISHED")) == 4
    assert len(selected_events(second, "CONTACT_PERSISTED")) == 4
    losses = selected_events(gone, "CONTACT_LOSS")
    assert len(losses) == 4
    e = losses[0]
    assert (e["run_id"], e["env_id"], e["episode_id"], e["physics_step"],
            e["control_step"], e["control_substep"], e["sim_time_s"]) == (
                "cpu-test", 0, 0, 2, 1, 0, .03)
    assert e["sampled_duration_s"] == pytest.approx(.02)
    assert e["start_physics_step"] == 0
    assert e["last_contact_physics_step"] == 1
    assert gone["reasons"]["CONTACT_LOSS"]
    assert records == [first, second, gone]
    assert events == first["events"] + second["events"] + gone["events"]
    json.dumps(d.finish(), allow_nan=False)


def test_episode_reset_censors_and_never_connects_same_pad_durations():
    d = reducer()
    d.observe(frame(0))
    record = d.observe(frame(1, row(episode=1)))[0]
    censored = selected_events(record, "CONTACT_CENSORED")
    assert len(censored) == 4
    assert all(e["episode_id"] == 0 and e["physics_step"] == 0 for e in censored)
    assert all(e["reason"] == "episode_changed" for e in censored)
    assert not record["reasons"]["CONTACT_LOSS"]
    assert all(e["sampled_duration_s"] == pytest.approx(.01)
               for e in selected_events(record, "CONTACT_ESTABLISHED"))


@pytest.mark.parametrize("invalid_kind", ["row", "pad"])
def test_unknown_sample_censors_without_loss_or_imputed_zero_slip(invalid_kind):
    d = reducer()
    d.observe(frame(0))
    r = row(contacts=False)
    if invalid_kind == "row":
        r["measurement_valid"] = False
    else:
        r["pads"][0]["valid"] = False
    record = d.observe(frame(1, r))[0]
    assert not record["source_validity"]["contact"]
    assert not record["reasons"]["NO_CONTACT"]
    assert len(selected_events(record, "CONTACT_CENSORED")) >= 1
    assert not any(e["digit"] == "thumb" and e["hand"] == "left"
                   for e in selected_events(record, "CONTACT_LOSS"))
    assert record["slip_m"] is None
    assert not record["source_validity"]["slip"]
    assert not record["reasons"]["SLIP"]
    resumed = d.observe(frame(2))[0]
    left_thumb = [e for e in selected_events(resumed, "CONTACT_ESTABLISHED")
                  if (e["hand"], e["digit"]) == ("left", "thumb")]
    assert left_thumb[0]["sampled_duration_s"] == pytest.approx(.01)


def test_terminal_and_finish_censor_active_contacts_with_last_sample_identity():
    events = []
    d = reducer(event_sink=events.append)
    terminal = row()
    terminal["done"] = True
    record = d.observe(frame(0, terminal))[0]
    assert len(selected_events(record, "CONTACT_CENSORED")) == 4
    assert all(e["reason"] == "episode_end" for e in selected_events(record, "CONTACT_CENSORED"))
    d.observe(frame(1, row(episode=1)))
    result = d.finish()
    assert result["finish_reason"] == "bounded_probe_end"
    assert len([e for e in events if e["event_type"] == "CONTACT_CENSORED"]) == 8
    assert not any(e["event_type"] == "CONTACT_LOSS" for e in events)
    assert d.finish() == result
    with pytest.raises(ValueError, match="finish"):
        d.observe(frame(2))


def test_reasons_are_independent_flags_and_native_success_is_only_echoed():
    d = reducer()
    d.observe(frame(0))
    r = row()
    for p in r["pads"]:
        if p["hand"] == "right":
            p.update(raw_load_n=0., load_n=0., accepted_load_n=0., contact_count=0,
                     position_w_m=None, normal_on_hand_w=None, raw_contact_ids=[])
        elif p["digit"] == "index":
            p["normal_on_hand_w"] = [1., 0., 0.]
        if p["digit"] == "thumb":
            p["wrong_normal_count"] = 1
    r.update(estimated_torque_nm=[.98] + [0.] * 11, slip_m=.0006,
             slip_valid=True, slip_applicable=True, slip_reason="measured")
    record = d.observe(frame(1, r))[0]
    assert set(record["reasons"]) == REASONS
    assert record["reasons"] == dict(NO_CONTACT=False, ONE_SIDE_CONTACT=True,
        WRONG_NORMAL=True, CONTACT_LOSS=True, NON_OPPOSED=True,
        TORQUE_LIMITED=True, SLIP=True)
    assert record["native_success"] is False


def test_wrong_normal_zero_accepted_and_strict_load_threshold():
    d = reducer()
    r = row()
    for p in r["pads"]:
        p.update(raw_load_n=1., load_n=.05, accepted_load_n=.05, wrong_normal_count=1)
    record = d.observe(frame(0, r))[0]
    assert record["reasons"]["NO_CONTACT"]
    assert record["reasons"]["WRONG_NORMAL"]
    assert not record["bilateral_opposed"]
    assert not record["events"]


def test_pinch_and_tripod_masks_do_not_silently_become_historical_all_five():
    d = reducer()
    a, b = row(env=0), row(env=1, style="tripod_thumb_index_middle")
    for r in (a, b):
        for p in r["pads"]:
            if p["digit"] in ("middle", "ring"):
                p.update(raw_load_n=1., load_n=1., accepted_load_n=1.,
                         normal_on_hand_w=[0., 1., 0.], position_w_m=[0., 0., 0.], contact_count=1)
    first, second = d.observe(frame(0, a, b))
    assert first["bilateral_opposed"]
    assert not second["bilateral_opposed"]
    contract = d.summary()["contract"]
    assert contract["style_masks"]["pinch_thumb_index"] == ["thumb", "index"]
    assert contract["historical_all_five_definition_unchanged"] is True


def test_tripod_uses_most_opposed_pair_then_requires_all_bearings_assigned():
    r = row(style="tripod_thumb_index_middle")
    for p in r["pads"]:
        if p["digit"] == "middle":
            p.update(raw_load_n=1., load_n=1., accepted_load_n=1., normal_on_hand_w=[-1., 0., 0.],
                     position_w_m=[0., 0., 0.], contact_count=1)
    assert reducer().observe(frame(0, r))[0]["bilateral_opposed"]
    r["pads"][2]["normal_on_hand_w"] = [0., 1., 0.]
    out = reducer().observe(frame(0, r))[0]
    assert not out["per_hand"]["left"]["opposed"]
    assert out["per_hand"]["right"]["opposed"]


def test_zero_normal_makes_opposition_unknown_instead_of_positive():
    r = row()
    r["pads"][0]["normal_on_hand_w"] = [0., 0., 0.]
    out = reducer().observe(frame(0, r))[0]
    assert out["per_hand"]["left"]["opposed"] is None
    assert not out["source_validity"]["opposition"]["bilateral"]
    assert not out["reasons"]["NON_OPPOSED"]


def test_style_duration_denominators_are_env_time_and_ignore_inactive_and_unknown():
    d = reducer()
    a, b = row(), row(env=1, style="tripod_thumb_index_middle")
    d.observe(frame(0, a, b))
    a = row(episode=1)
    a["measurement_valid"] = False
    b["live"] = False
    d.observe(frame(1, a, b))
    s = d.summary()["by_style"]
    pinch = s["pinch_thumb_index"]
    assert pinch["live_s"] == pytest.approx(.02)
    assert pinch["valid_s"] == pytest.approx(.01)
    assert pinch["coverage"] == pytest.approx(.5)
    assert pinch["bilateral"]["opposed_s"] == pytest.approx(.01)
    assert pinch["bilateral"]["opposed_sample_ratio"] == 1.
    assert pinch["bilateral"]["coverage"] == .5
    assert s["tripod_thumb_index_middle"]["live_s"] == pytest.approx(.01)
    assert d.summary()["episodes_seen"] == 3


@pytest.mark.parametrize("mutation", ["partial", "source", "missing"])
def test_torque_reason_requires_all_twelve_valid_simulation_estimates(mutation):
    r = row()
    r["estimated_torque_nm"][0] = -1.
    if mutation == "partial":
        r["torque_valid"][11] = False
    elif mutation == "source":
        r["torque_source"] = None
    else:
        r.update(estimated_torque_nm=None, effort_limit_nm=None, torque_valid=None, torque_source=None)
    out = reducer().observe(frame(0, r))[0]
    assert not out["source_validity"]["torque"]
    assert not out["reasons"]["TORQUE_LIMITED"]


@pytest.mark.parametrize("value,valid,applicable,expected", [(.0005, True, True, False),
    (.0005001, True, True, True), (.001, False, True, False), (.001, True, False, False)])
def test_slip_only_uses_valid_applicable_strict_native_distance_threshold(value, valid, applicable, expected):
    r = row()
    r.update(slip_m=value, slip_valid=valid, slip_applicable=applicable, slip_reason="fixture")
    out = reducer().observe(frame(0, r))[0]
    assert out["reasons"]["SLIP"] is expected
    summary = reducer().summary()
    assert summary["by_style"][r["grasp_style"]]["slip"]["coverage"] is None


@pytest.mark.parametrize("mutation", ["repeat_step", "time_back", "repeat_control", "negative_dt"])
def test_rejects_nonmonotone_frame_identity_before_streaming(mutation):
    records = []
    d = reducer(sink=records.append)
    d.observe(frame(0))
    bad = frame(1)
    if mutation == "repeat_step": bad["physics_step"] = 0
    elif mutation == "time_back": bad["sim_time_s"] = .005
    elif mutation == "repeat_control": bad.update(control_step=0, control_substep=0)
    else: bad["dt_s"] = -.01
    with pytest.raises(ValueError): d.observe(bad)
    assert len(records) == 1


@pytest.mark.parametrize("mutation", ["duplicate_row", "duplicate_pad", "missing_pad", "style_change",
    "episode_back", "body_change", "env_change", "nonfinite_load", "nonfinite_normal", "bad_torque_length",
    "nonfinite_torque", "negative_effort", "nonfinite_raw", "bad_bool", "negative_slip"])
def test_strict_schema_and_identity_reject_bad_frames_atomically(mutation):
    records = []
    d = reducer(sink=records.append)
    d.observe(frame(0))
    r = row()
    bad = frame(1, r)
    if mutation == "duplicate_row": bad["rows"].append(copy.deepcopy(r))
    elif mutation == "duplicate_pad": r["pads"][1] = copy.deepcopy(r["pads"][0])
    elif mutation == "missing_pad": r["pads"].pop()
    elif mutation == "style_change": r["grasp_style"] = "tripod_thumb_index_middle"
    elif mutation == "episode_back": r["episode_id"] = -1
    elif mutation == "body_change": r["pads"][0]["body_id"] = 900
    elif mutation == "env_change": r["env_id"] = 10
    elif mutation == "nonfinite_load": r["pads"][0]["accepted_load_n"] = math.nan
    elif mutation == "nonfinite_normal": r["pads"][0]["normal_on_hand_w"][0] = math.inf
    elif mutation == "bad_torque_length": r["torque_valid"].pop()
    elif mutation == "nonfinite_torque": r["estimated_torque_nm"][0] = math.inf
    elif mutation == "negative_effort": r["effort_limit_nm"][0] = -1.
    elif mutation == "nonfinite_raw": r["raw_contacts"] = [{"separation_m": math.nan}]
    elif mutation == "bad_bool": r["native_success"] = 1
    else: r["slip_m"] = -.01
    with pytest.raises(ValueError): d.observe(bad)
    assert len(records) == 1
    assert d.summary()["by_style"]["pinch_thumb_index"]["live_s"] == pytest.approx(.01)


def test_raw_records_are_snapshot_copies_and_heap_retains_no_raw_history():
    records = []
    d = reducer(sink=records.append)
    f = frame()
    f["rows"][0]["raw_contacts"] = [{"point_w_m": [1., 2., 3.], "normal_w": [1., 0., 0.]}]
    out = d.observe(f)[0]
    f["rows"][0]["pads"][0]["accepted_load_n"] = 999.
    f["rows"][0]["raw_contacts"][0]["point_w_m"][0] = 999.
    assert out["pads"][0]["accepted_load_n"] == 1.
    assert records[0]["raw_contacts"][0]["point_w_m"] == [1., 2., 3.]
    for step in range(1, 100): d.observe(frame(step))
    assert not any(isinstance(v, list) and len(v) >= 100 for v in vars(d).values())
    json.dumps(d.summary(), allow_nan=False)


def test_missing_physics_sample_breaks_continuity_without_claiming_loss():
    d = reducer()
    d.observe(frame(0))
    record = d.observe(frame(3))[0]
    assert len(selected_events(record, "CONTACT_CENSORED")) == 4
    assert all(e["reason"] == "sample_gap" for e in selected_events(record, "CONTACT_CENSORED"))
    assert len(selected_events(record, "CONTACT_ESTABLISHED")) == 4
    assert not selected_events(record, "CONTACT_PERSISTED")
    assert not record["reasons"]["CONTACT_LOSS"]


def test_known_wrong_normal_remains_independent_of_other_missing_pad_measurements():
    r = row()
    r["pads"][0]["wrong_normal_count"] = 1
    r["pads"][1]["valid"] = False
    out = reducer().observe(frame(0, r))[0]
    assert out["reasons"]["WRONG_NORMAL"]
    assert out["reason_validity"]["WRONG_NORMAL"]
    assert not out["source_validity"]["contact"]


def test_contact_loss_negative_is_observable_with_complete_contact_measurements():
    out = reducer().observe(frame(0, row(contacts=False)))[0]
    assert out["reason_validity"]["CONTACT_LOSS"]
    assert not out["reasons"]["CONTACT_LOSS"]


def test_surface_opposition_matches_existing_all_five_function_with_style_load_mask():
    torch = pytest.importorskip("torch")
    from go2_pvcnn.control.dh116.tabletop_surface_force_sharing import surface_targets
    import random
    random_generator = random.Random(116)
    for style in ("pinch_thumb_index", "tripod_thumb_index_middle"):
        mask = {"thumb", "index"} if style == "pinch_thumb_index" else {"thumb", "index", "middle"}
        for _ in range(30):
            r = row(style=style)
            loads, normals = [], []
            for hand in HANDS:
                hloads, hnormals = [], []
                for p in (p for p in r["pads"] if p["hand"] == hand):
                    load = random_generator.choice([0., .05, .1, 1.])
                    normal = [random_generator.uniform(-1., 1.) for _ in range(3)]
                    p.update(raw_load_n=load, load_n=load, accepted_load_n=load,
                             normal_on_hand_w=normal)
                    hloads.append(load if p["digit"] in mask else 0.)
                    hnormals.append(normal)
                loads.append(hloads)
                normals.append(hnormals)
            expected = surface_targets(torch.tensor(loads, dtype=torch.float64),
                                       torch.tensor(normals, dtype=torch.float64))["paired"].tolist()
            actual = reducer().observe(frame(0, r))[0]
            assert [actual["per_hand"][h]["opposed"] for h in HANDS] == expected


@pytest.mark.parametrize("field", ["load_n", "accepted_load_n"])
def test_pad_gates_cannot_increase_positive_loads(field):
    r = row()
    r["pads"][0][field] = 2.
    with pytest.raises(ValueError, match="load"):
        reducer().observe(frame(0, r))


def test_explicitly_finished_env_can_be_omitted_without_changing_active_env_identity():
    d = reducer()
    a, b = row(), row(env=1)
    a["done"] = True
    d.observe(frame(0, a, b))
    out = d.observe(frame(1, b))
    assert [r["env_id"] for r in out] == [1]
    assert not out[0]["reasons"]["CONTACT_LOSS"]
    assert d.summary()["episodes_seen"] == 2
    with pytest.raises(ValueError):
        d.observe(frame(2, row(env=2)))


def test_unknown_body_binding_is_retained_as_unknown_then_resolves_once():
    d = reducer()
    r = row()
    r["pads"][0].update(body_id=-1, valid=False, accepted_load_n=0.)
    out = d.observe(frame(0, r))[0]
    assert out["pads"][0]["effective_contact"] is None
    assert not out["source_validity"]["contact"]
    out = d.observe(frame(1))[0]
    assert any(e["body_id"] == 0 and e["hand"] == "left" and e["digit"] == "thumb"
               for e in selected_events(out, "CONTACT_ESTABLISHED"))
    r["pads"][0].update(body_id=-1, valid=False)
    out = d.observe(frame(2, r))[0]
    e = [e for e in selected_events(out, "CONTACT_CENSORED") if e["hand"] == "left"][0]
    assert e["body_id"] == 0
    r = row()
    r["pads"][0]["body_id"] = 900
    with pytest.raises(ValueError): d.observe(frame(3, r))


def test_unknown_body_cannot_be_marked_valid():
    r = row()
    r["pads"][0]["body_id"] = -1
    with pytest.raises(ValueError): reducer().observe(frame(0, r))


@pytest.mark.parametrize("kind", ["compression", "wrong_normal", "nonfinite", "signed_release"])
def test_native_adapter_to_reducer_keeps_unknown_and_raw_provenance(kind):
    torch = pytest.importorskip("torch")
    native = importlib.import_module("go2_pvcnn.control.dh116.tabletop_grasp_contact_native")
    poses = torch.tensor([[[0., 0., 0., 1., 0., 0., 0.]]], dtype=torch.float64)
    patch = dict(center_local_m=[0., 0., 0.], tangent_u=[1., 0., 0.], tangent_v=[0., 0., 1.],
                 half_widths_m=[.004, .0035], planes_local=[[0., -1., 0., 0.]],
                 surface_tolerance_m=.001, normal_cosine_min=.8)
    force = -.2 if kind == "signed_release" else .2
    normal = [0., -1., 0.] if kind == "wrong_normal" else [0., 1., 0.]
    point = [math.nan, 0., 0.] if kind == "nonfinite" else [0., 0., 0.]
    raw = (torch.tensor([force]), torch.tensor([point]), torch.tensor([normal]),
           torch.tensor([-.0001]), torch.tensor([1]), torch.tensor([0]))
    measured = native.reduce_pad_contact_buffers([(0, 0, 0)], [raw], [poses, poses], {(0, 0): patch}, 1)[0]
    r = row()
    r.update(measured)
    out = reducer().observe(frame(0, r))[0]
    assert out["pads"][1]["body_id"] == -1
    assert out["pads"][1]["effective_contact"] is None
    assert not out["source_validity"]["contact"]
    assert not out["reasons"]["NO_CONTACT"]
    assert out["reasons"]["WRONG_NORMAL"] is (kind == "wrong_normal")
    assert bool(selected_events(out, "CONTACT_ESTABLISHED")) is (kind == "compression")
    raw_record = out["raw_contacts"][0]
    assert raw_record["normal_scalar_n"] == pytest.approx(force)
    assert raw_record["normal_on_hand_w"] == normal
    assert raw_record["position_w_m"][0] is None if kind == "nonfinite" else raw_record["position_w_m"] == point
    assert out["pads"][0]["normal_source"] == "reduced_native_force_direction"
    json.dumps(out, allow_nan=False)


def test_terminal_episode_identity_stays_closed_through_inactive_rows():
    d = reducer()
    r = row()
    r["done"] = True
    d.observe(frame(0, r))
    r = row()
    r.update(live=False, done=False)
    d.observe(frame(1, r))
    with pytest.raises(ValueError, match="reset"):
        d.observe(frame(2))


def test_style_loss_reason_ignores_optional_digits_but_retains_all_pad_events():
    d = reducer()
    r = row()
    r["pads"][3].update(raw_load_n=1., load_n=1., accepted_load_n=1.,
                        position_w_m=[0., 0., 0.], normal_on_hand_w=[0., 1., 0.], contact_count=1)
    d.observe(frame(0, r))
    out = d.observe(frame(1))[0]
    losses = selected_events(out, "CONTACT_LOSS")
    assert [(e["hand"], e["digit"]) for e in losses] == [("left", "ring")]
    assert not out["reasons"]["CONTACT_LOSS"]
    assert d.summary()["by_style"]["pinch_thumb_index"]["pads"]["left"]["ring"]["lost"] == 1
    assert d.summary()["contract"]["reason_digit_scope"] == "grasp_style_mask"
    assert d.summary()["contract"]["pad_event_digit_scope"] == "all_ten_pads"
