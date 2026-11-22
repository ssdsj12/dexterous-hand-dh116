import numpy as np
import pytest

from scripts.verify_dh116_supported_force_seating import validate_seating_schedule, select_seating_command


def evidence(index=.06, middle=.03, loaded=False, total=.32):
    pads = {"thumb": .11, "index": index, "middle": middle}
    return {"per_hand": {"right": {"loaded": loaded, "per_substep": [
        {"per_pad": {pad: {"normal_load_n": force} for pad, force in pads.items()}, "total_normal_load_n": total}
        for _ in range(12)]}}}


def test_only_two_screened_fingers_can_change_at_fixed_wrist():
    command = np.zeros(24)
    command[[8, 9]] = .06
    validate_seating_schedule(np.stack((command, np.zeros(24))), {8: "index", 9: "middle"}, max_travel_rad=.003)
    issued, limits = select_seating_command(command, evidence(index=.10), {8: "index", 9: "middle"})
    assert limits == {"index": .10, "middle": .03}
    assert np.flatnonzero(issued).tolist() == [9]
    continued, _ = select_seating_command(command, evidence(index=.09, middle=.09, total=.28), {8: "index", 9: "middle"})
    np.testing.assert_array_equal(continued, command)
    np.testing.assert_array_equal(select_seating_command(command, evidence(loaded=True), {8: "index", 9: "middle"})[0],
                                  np.zeros(24))


@pytest.mark.parametrize("slot", [1, 7, 10, 12, 16, 21])
def test_tentative_seating_rejects_any_other_action_channel(slot):
    command = np.zeros((2, 24))
    command[:, slot] = .01
    with pytest.raises(ValueError, match="finger-only"):
        validate_seating_schedule(command, {8: "index", 9: "middle"}, max_travel_rad=.003)


def test_budget_and_same_sample_force_completeness():
    command = np.zeros((11, 24))
    command[:, 9] = .06
    with pytest.raises(ValueError, match="travel"):
        validate_seating_schedule(command, {9: "middle"}, max_travel_rad=.003)
    reading = evidence()
    reading["per_hand"]["right"]["per_substep"].pop()
    with pytest.raises(ValueError, match="12"):
        select_seating_command(command[0], reading, {9: "middle"})


def test_pinch_seating_uses_only_the_right_index_and_its_own_receipt(monkeypatch, tmp_path):
    from scripts import verify_dh116_supported_force_seating as seating

    repo = tmp_path
    request = repo / "configs/dh116_tasks/flexible_pinch_right_force_seating_v1.json"
    request.parent.mkdir(parents=True)
    import json
    request.write_text(json.dumps(dict(version="dh116-pinch-right-force-seating-v1",control_boundary_sdk=656,
        maximum_finger_travel_rad=.0012,planned_index_action=.03,dose_windows=8,settle_windows=16)))
    monkeypatch.setattr(seating.measured, "ROOT", repo)
    sealed = {"physics_step": 656, "source": "Pinch original right"}
    receipt = repo / "receipt.json"

    def archive(path, *, contract, sdk, task):
        assert path == receipt
        assert sdk == 656 and task == "pinch"
        assert contract.name == "dh116_teacher_pinch_right_first_contact_native_v1.json"
        return [sealed], [], (receipt,)

    monkeypatch.setattr(seating.measured, "archive", archive)
    commands, plan, sources = seating.load_pinch_seating(receipt)
    assert commands.shape == (24, 24)
    assert np.flatnonzero(np.any(commands,axis=0)).tolist() == [8]
    np.testing.assert_allclose(commands[:8,8], .03)
    np.testing.assert_array_equal(commands[8:], 0.)
    assert plan["archive"] == sealed and plan["pad_by_slot"] == {"8":"index"} and receipt in sources


def test_left_index_seating_reads_left_force_and_preserves_right_channel():
    command=np.zeros((4,24))
    command[:,2]=.03
    validate_seating_schedule(command,{2:"index"},max_travel_rad=.0006)
    right_only=evidence(index=.10,loaded=True)
    left_only=evidence(index=.01,loaded=False)["per_hand"]["right"]
    right_only["per_hand"]["left"]=left_only
    actual, force=select_seating_command(command[0],right_only,{2:"index"},side="left")
    assert np.flatnonzero(actual).tolist()==[2]
    assert force=={"index":.01}
    with pytest.raises(ValueError,match="measured hand identity"):
        select_seating_command(command[0],right_only,{2:"index"},side="right")


def test_tripod_replay_stage_emits_only_its_own_named_slots(monkeypatch, tmp_path):
    from scripts import verify_dh116_supported_force_seating as seating

    import json
    receipt = tmp_path / "outer.json"
    path = tmp_path / "native" / "tripod_control_1296.json"
    path.parent.mkdir()
    receipt.write_text(json.dumps({"artifact_sha256": {path.name:"fixture"}}))
    path.write_text(json.dumps({"physics_step":1296,"sdk_hand_geometry":{"physics_step":1296},
        "rod":{"physics_step":1296,"rod_tick":15528},"guard_failure":None}))
    contract=tmp_path/"contract.json"
    contract.write_text(json.dumps({"actual":{"outer_sha256":"fixture","end_sdk":1456},
                                    "force_followup_boundary":{"sdk":1296}}))
    monkeypatch.setattr(seating,"TRIPOD_CONTRACT",contract)
    monkeypatch.setattr(seating.measured.sources,"checked_json",lambda p,d:json.loads(p.read_bytes()))
    monkeypatch.setattr(seating.measured,"archive",lambda *args,**kwargs:([{"physics_step":1456}],[],(receipt,)))
    cmd,stage,_=seating.load_tripod_seating(receipt,np.zeros((152,24)),start_sdk=240)
    planned=cmd[(1296-240)//8]
    pads={int(slot):pad for slot,pad in stage["pad_by_slot"].items()}
    assert pads=={8:"index",9:"middle"}
    assert select_seating_command(planned,evidence(),pads)[0].nonzero()[0].tolist()==[8,9]


def bilateral_evidence(left=(.319, .235, .133), right=(.109, .056, .033), *, sdk=1296):
    reading = dict(grasp="Tripod", solver_substeps=12,
                   identity=dict(run_uuid="test-run", env_id=0, episode_uid="test-episode",
                                 reset_id=1, rod_uid="test-rod", physics_step=sdk, rod_tick=(sdk-2)*12),
                   wrong_side_contacts=[], unexpected_hand_contacts=[], per_hand={})
    for side, loads in (("left", left), ("right", right)):
        pads = dict(zip(("thumb", "index", "middle"), loads, strict=True))
        loaded = min(loads) > .05 and sum(loads) > .3
        reading["per_hand"][side] = dict(loaded=loaded, required_pads=list(pads), per_substep=[
            dict(substep=i, loaded=loaded, opposition_cos=-.5, effective=True,
                 all_required_pads_on_opposing_faces=True, total_normal_load_n=sum(loads),
                 per_pad={name:dict(normal_load_n=value) for name,value in pads.items()})
            for i in range(12)])
    return reading


def bilateral_controller(**kwargs):
    from scripts.verify_dh116_supported_force_seating import BilateralLoadFeedback
    limits = kwargs.pop("max_travel_rad", {1:0.,2:.0006,3:.0012,7:0.,8:.0012,9:.0018})
    return BilateralLoadFeedback(start_sdk=1296, windows=8, action_step=.03,
        pad_target_n=.085, max_travel_rad=limits, **kwargs)


def test_bilateral_feedback_preserves_loaded_hand_and_selects_weakest_pad():
    controller = bilateral_controller()
    command, report = controller.decide(bilateral_evidence(), sdk=1296)
    assert np.flatnonzero(command).tolist() == [9]
    assert command[9] == pytest.approx(.03)
    assert report["hands"]["left"]["reason"] == "loaded_hold"
    assert report["hands"]["right"]["selected_pad"] == "middle"
    assert not report["grasp_qualified"] and not report["teacher_admission"]
    # A left load loss is handled next window, without waiting for a separate left stage.
    command, report = controller.decide(bilateral_evidence(left=(.20,.02,.12), sdk=1304), sdk=1304)
    assert np.flatnonzero(command).tolist() == [2,9]
    np.testing.assert_array_equal(command[12:], 0.)


def test_bilateral_total_force_deficit_is_not_hidden_by_individual_pad_threshold():
    controller = bilateral_controller(max_travel_rad={1:0.,2:.0006,3:.0012,7:.0003,8:.0012,9:.0018})
    command, report = controller.decide(bilateral_evidence(right=(.11,.09,.09)), sdk=1296)
    assert np.flatnonzero(command).tolist() == [7]  # weaker opposing side, not all three
    assert report["hands"]["right"]["minimum_total_load_n"] == pytest.approx(.29)


def test_bilateral_exhaustion_does_not_redirect_force_into_an_already_loaded_pad():
    controller = bilateral_controller()
    for i in range(8):
        command, report = controller.decide(bilateral_evidence(right=(.20,.01,.15),sdk=1296+8*i),sdk=1296+8*i)
        assert np.flatnonzero(command).tolist() == [8]
    with pytest.raises(ValueError, match="window"):
        controller.decide(bilateral_evidence(sdk=1360), sdk=1360)
    # Smaller per-joint limit is consumed exactly; no replacement with another, stronger pad.
    controller = bilateral_controller(max_travel_rad={1:0.,2:.0006,3:.0012,7:0.,8:.00015,9:.0018})
    controller.decide(bilateral_evidence(right=(.20,.01,.15)), sdk=1296)
    command, report = controller.decide(bilateral_evidence(right=(.20,.01,.15),sdk=1304), sdk=1304)
    assert not command.any()
    assert report["hands"]["right"]["reason"] == "weakest_pad_travel_exhausted"


@pytest.mark.parametrize("damage", ["stale", "duplicate_substep", "missing_pad", "nan_force", "wrong_side", "reset"])
def test_bilateral_feedback_rejects_unknown_or_wrong_identity_without_spending_travel(damage):
    controller = bilateral_controller()
    controller.decide(bilateral_evidence(), sdk=1296)
    saved = controller.travel_rad.copy()
    reading = bilateral_evidence(sdk=1304)
    if damage == "stale": reading["identity"]["physics_step"] = 1296
    if damage == "duplicate_substep": reading["per_hand"]["right"]["per_substep"][-1]["substep"] = 0
    if damage == "missing_pad": del reading["per_hand"]["right"]["per_substep"][0]["per_pad"]["index"]
    if damage == "nan_force": reading["per_hand"]["right"]["per_substep"][0]["per_pad"]["index"]["normal_load_n"] = float("nan")
    if damage == "wrong_side": reading["wrong_side_contacts"] = [dict(link="right.finger22")]
    if damage == "reset": reading["identity"]["reset_id"] += 1
    with pytest.raises(ValueError):
        controller.decide(reading, sdk=1304)
    np.testing.assert_array_equal(controller.travel_rad, saved)


def test_bilateral_feedback_does_not_close_blindly_when_only_opposition_is_missing():
    controller = bilateral_controller()
    reading = bilateral_evidence(right=(.2,.2,.2))
    reading["per_hand"]["right"]["loaded"] = False
    for row in reading["per_hand"]["right"]["per_substep"]:
        row["loaded"] = False
        row["opposition_cos"] = .4
    command, report = controller.decide(reading, sdk=1296)
    assert not command.any()
    assert report["hands"]["right"]["reason"] == "opposition_or_geometry_requires_retarget"


def test_bilateral_loader_keeps_exact_anchor_and_replaces_old_serial_schedule(monkeypatch):
    from scripts import verify_dh116_supported_force_seating as seating
    from pathlib import Path

    request = Path(seating.__file__).resolve().parents[1] / "configs/dh116_tasks/flexible_tripod_bilateral_load_feedback_v1.json"
    anchor = {"physics_step":1296}
    source = Path("actual-receipt.json")
    def replay(receipt, actions, *, start_sdk):
        assert receipt == source and start_sdk == 240
        return actions, dict(sdk_step=1296,archive=anchor,pad_by_slot={"8":"index","9":"middle"}), (source,)
    monkeypatch.setattr(seating, "load_tripod_seating", replay)
    prefix = np.arange(152*24, dtype=float).reshape(152,24)
    schedule, stage, files = seating.load_tripod_bilateral_feedback(source,prefix,start_sdk=240,request_path=request)
    np.testing.assert_array_equal(schedule[:132],prefix[:132])
    np.testing.assert_array_equal(schedule[132:],np.zeros((64,24)))
    assert stage["archive"] is anchor and stage["variant"] == "bilateral_load_feedback_v1"
    assert stage["stop_sdk"] == 1808 and request in files and source in files
    import json
    assert json.loads(json.dumps(stage)) == stage
    seating.BilateralLoadFeedback(**json.loads(json.dumps(stage))["feedback"])


@pytest.mark.parametrize("variant,windows,error", [
    ("bilateral_load_feedback_v1",1,None),
    ("unknown_feedback",1,"variant"),
    ("bilateral_load_feedback_v1",2,"consumed"),
])
def test_real_supported_action_hook_issues_feedback_before_physical_control(monkeypatch, tmp_path,variant,windows,error):
    from types import SimpleNamespace
    from scripts import verify_dh116_power_shared_rod as shared
    from go2_pvcnn.control.dh116 import tabletop_deformable_physical_exchange as exchange
    from go2_pvcnn.control.dh116 import tabletop_power_contact_evidence as contacts
    from go2_pvcnn.control.dh116.tabletop_power_shared_rod import PALM_NAMES

    actual_commands = []
    written = {}
    power = SimpleNamespace(sdk=1296, _base=SimpleNamespace(last={"physics_step":1296}))
    power.clock = lambda: {"physics_step":power.sdk}
    probe = SimpleNamespace(previous={"physics_step":1296,"rod":power._base.last},
        bilateral=dict(boundary_sdk=1136,midpoint_sdk=1000,seating=dict(sdk_step=1296,stop_sdk=1296+8*windows,variant=variant,
            feedback=dict(start_sdk=1296,windows=windows,action_step=.03,pad_target_n=.085,
                          max_travel_rad={1:0.,2:.0006,3:.0012,7:0.,8:.0012,9:.0018}))),
        archive=None)
    def observe(row):
        probe.previous = row
        return {}, None
    probe.observe = observe
    probe.active = lambda sdk: True
    probe.summary = lambda: {"diagnostic_only":True}
    poses = {name:(np.zeros(3),np.array([0.,0.,0.,1.])) for name in PALM_NAMES}
    monkeypatch.setattr(shared, "measure_links", lambda *a,**k:(poses,{},np.array([.01,.01])))
    monkeypatch.setattr(shared.native, "clock", lambda sim:{"sdk_step":power.sdk})
    monkeypatch.setattr(shared.native, "write_new", lambda path,value:written.setdefault(path.name,value))
    monkeypatch.setattr(contacts,"measure_supported_distal_contact",
                        lambda rod,*,grasp:bilateral_evidence(sdk=rod["physics_step"]))

    class PhysicalDriver:
        def __init__(self, sim, rod, bindings, **kwargs):
            self.after = kwargs["read_geometry_after_sdk"]
        def control(self, action, *, observe):
            actual_commands.append(action.numpy().copy())
            for _ in range(8):
                previous = power._base.last
                power.sdk += 1
                self.after(previous)
                power._base.last = {"physics_step":power.sdk}
                assert observe({"physics_step":power.sdk,"rod":power._base.last}) is None
            return dict(physical_samples=8,termination_reason=None)
    monkeypatch.setattr(exchange,"FlexiblePhysicalExchange",PhysicalDriver)
    owner=SimpleNamespace(keep=lambda name,obj:obj)
    trace=SimpleNamespace(watch=lambda *a,**k:None)
    setup={"root_pose_wxyz":[[0.,0.,0.,1.,0.,0.,0.]]*2}
    def run():
        return shared.run_supported_action(object(),owner,trace,tmp_path,power=power,
            bindings={"left":object(),"right":object()},meshes={},clearance_shapes={},setup=setup,
            action=np.zeros((1,24)),sdk_end=1304,task="tripod",distal_probe=probe)
    if error is not None:
        with pytest.raises(ValueError,match=error):
            run()
        if variant == "unknown_feedback":
            assert not actual_commands
        return
    result=run()
    assert result["physical_samples"] == 8 and power.sdk == 1304
    assert np.flatnonzero(actual_commands[0][0]).tolist() == [9]
    decision=written["tripod_bilateral_command_1296.json"]
    assert decision["source_sdk_step"] == 1296
    assert decision["issued_action"][9] == pytest.approx(.03)
    assert decision["hands"]["left"]["reason"] == "loaded_hold"
    assert len([name for name in written if name.startswith("tripod_control_") and name != "tripod_control_result.json"]) == 8


def test_bilateral_feedback_cannot_freeze_a_hand_on_a_forged_loaded_boolean():
    controller = bilateral_controller()
    reading = bilateral_evidence()
    reading["per_hand"]["right"]["loaded"] = True
    for row in reading["per_hand"]["right"]["per_substep"]:
        row["loaded"] = True
    with pytest.raises(ValueError, match="loaded"):
        controller.decide(reading,sdk=1296)
    assert not controller.travel_rad.any()


def test_bilateral_configuration_rejects_boolean_travel_instead_of_silently_disabling_a_finger():
    with pytest.raises(ValueError,match="travel"):
        bilateral_controller(max_travel_rad={1:0.,2:False,3:.0012,7:0.,8:.0012,9:.0018})
