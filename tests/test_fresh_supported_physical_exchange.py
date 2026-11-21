"""Fresh seven-route exchange from committed17-slot Reset; CPU doubles only."""
from copy import deepcopy
import numpy as np
import pytest
import torch

from test_supported_mesh_reset import case, execute
from test_dh116_deformable_bimanual_step import Hand as ActionHand
from go2_pvcnn.control.dh116.tabletop_deformable_physical_exchange import FlexiblePhysicalExchange


def prepared(monkeypatch):
    # Construct41 material stations in the fake backend from the outset.
    nodes = np.c_[np.zeros(41), np.linspace(-.25,.25,41), np.full(41,.31)]
    c, module, plan = case(monkeypatch,nodes=nodes)
    _, receipt = execute(c,module,plan)
    c.calls, c.first_solver_from = [], None
    for hand in c.hands:
        # Same composer owner; implement the native indexed writer contract
        # absent from the original Reset-only CPU fixture.
        composer = hand.permanent_wrench_composer
        def write(*,forces,torques,body_ids=None,is_global=False,hand=hand,composer=composer):
            if body_ids is None:
                composer.composed_force_as_torch[:] = forces
                composer.composed_torque_as_torch[:] = torques
            else:
                composer.composed_force_as_torch[:,body_ids] = forces
                composer.composed_torque_as_torch[:,body_ids] = torques
            composer.active = True
            c.calls.append((hand.side, None if body_ids is None else list(body_ids), forces.clone(), torques.clone(), is_global))
        composer.set_forces_and_torques = write
        hand.write_data_to_sim = lambda hand=hand: ActionHand.write_data_to_sim(hand)
    c.sim.is_playing = lambda: True
    def sdk_step(*,render):
        assert render is False
        c.sim.current_time_step_index += 1
        c.sim.current_time += 1/480
        # Explicit CPU SDK movement, not asserted to be a real physical response.
        c.hands[0].link_poses[0,0,0] += .00001
    c.sim.step = sdk_step
    step = c.rope.step_free
    def solver(dt,substeps):
        step(dt,substeps)
        if c.first_solver_from is None:
            c.first_solver_from = deepcopy(c.rope.started_from)
        c.rope.force[0,0,0] = .2
        c.rope.wrenches[0,16,0] = -.2
    c.rope.step_free = solver
    def pairs():
        point = c.hands[1].link_poses[0,6,:3].numpy()
        return dict(substeps=12,mesh_count=17,capacity=41*17*12,attempted=12,overflow=False,
            indices=np.array([[i,0,16] for i in range(12)],np.int32),
            vectors=np.array([[*point,1.,0.,0.,.2,0.,0.,0.,0.,0.] for _ in range(12)]),
            node_forces_on_rod_world_n=c.rope.force[0].copy(),
            mesh_wrenches_on_mesh_world=c.rope.wrenches[0].copy())
    c.rope.contact_pair_samples = pairs
    def geometry():
        poses = {}
        for hand in c.hands:
            for name, binding in c.rod.bindings.items():
                if binding["side"] == hand.side:
                    pose = hand.link_poses[0,binding["body_index"]].numpy()
                    poses[name] = (pose[:3].copy(),pose[3:].copy())
        return dict(physics_step=c.sim.current_time_step_index,
            root_poses_w=torch.stack([h.data.root_pose_w[0] for h in c.hands])[None],
            clearance_m=torch.full((1,2),.012,dtype=torch.float64),
            pad_poses=poses,centers_of_mass={key:value[0].copy() for key,value in poses.items()})
    c.read = geometry
    c.seed_tensor = torch.tensor(receipt["seed"]["reset_root_pose_w"],dtype=torch.float64)[None]
    return c,module,plan,receipt


def attach(c,plan,receipt,**kw):
    return FlexiblePhysicalExchange(c.sim,c.rod,c.bindings,root_seed_w=c.seed_tensor,
        read_geometry=c.read,physics_budget_end=18,
        measured_supported_reset=receipt,supported_mesh_plan=plan,**kw)


def test_fresh17_first_action_uses_committed_reset_then_delivers_prior_seven_route_feedback(monkeypatch):
    c,module,plan,receipt = prepared(monkeypatch)
    before = (c.sim.current_time_step_index,c.rope.time,c.rope.steps,c.rope.resets)
    exchange = attach(c,plan,receipt)
    assert before == (c.sim.current_time_step_index,c.rope.time,c.rope.steps,c.rope.resets)
    assert not c.calls
    with pytest.raises(ValueError,match="attached|claimed"):
        attach(c,plan,receipt)
    rows = []
    action = torch.zeros(1,24,dtype=torch.float64)
    action[0,12] = .02
    report = exchange.control(action,observe=lambda row:rows.append(row))
    assert report["physical_samples"] == 8 and report["termination_reason"] is None
    assert c.sim.current_time_step_index == 10 and c.rope.steps == 8
    assert [r["physics_step"] for r in rows] == list(range(3,11))
    assert [r["reaction_source_physics_step"] for r in rows] == [None,*range(3,10)]
    for side in ("left","right"):
        first = rows[0]["issued_composer_wrench"][side]
        assert first["forces"].shape == first["torques"].shape == (1,7,3)
        assert not first["forces"].any() and len(first["body_ids"]) == 7
        for previous,current in zip(rows,rows[1:]):
            np.testing.assert_array_equal(current["issued_composer_wrench"][side]["forces"],
                                          previous["rod"]["hand_wrenches_com_world"][side]["forces"])
    assert rows[0]["issued_composer_wrench_source"] == "commanded_reset_zero_not_measured_contact"
    assert all(row["issued_composer_wrench_source"] == "previous_measured_rod_step" for row in rows[1:])
    assert rows[1]["issued_composer_wrench"]["right"]["forces"][0,6,0] == pytest.approx(-.2)
    np.testing.assert_array_equal(c.first_solver_from[0][0],
        np.asarray(receipt["pad_poses_xyzw"]["left.finger14_link"][0],np.float32))
    assert not np.array_equal(c.rope.current[0][0],c.first_solver_from[0][0])
    assert all(np.array_equal(r["action_readback"]["request"]["action24"],action.numpy()) for r in rows)
    assert report["teacher_admission"] is False and not any(r["native_qualification"] for r in rows)


def test_fresh17_rejects_foreign_or_pending_readback_and_live_mesh_drift_before_writes(monkeypatch):
    c,module,plan,receipt = prepared(monkeypatch)
    for fault in ("owner","pending","live_mesh_drift"):
        row = deepcopy(receipt)
        original = c.rope.current[16][0].copy()
        if fault == "owner": row["mesh_state_readback"]["after"]["source_instance_id"] += 1
        elif fault == "pending": row["mesh_state_readback"]["after"]["snapshot"]["pending_set"] = True
        else: c.rope.current[16][0][0] += .001
        try:
            with pytest.raises(ValueError):
                attach(c,plan,row)
        finally:
            c.rope.current[16][0][:] = original
        assert not c.calls and c.sim.current_time_step_index == 2 and c.rope.steps == 0
        assert not getattr(c.rod,"_action_exchange_claimed",False)


def test_fresh17_and_measured_continuation_are_mutually_exclusive(monkeypatch):
    c,module,plan,receipt = prepared(monkeypatch)
    with pytest.raises(ValueError,match="exclusive|both"):
        attach(c,plan,receipt,continue_supported_rod=True)
    assert not c.calls and c.rope.steps == 0


def test_new_verified_reset_allows_a_new_lease_and_invalidates_the_old_exchange(monkeypatch):
    c,module,plan,receipt = prepared(monkeypatch)
    first = attach(c,plan,receipt)
    first.control(torch.zeros(1,24),observe=lambda row: None)
    second = module.MeasuredSupportedMeshReset(c.seed,episode_uid="cpu17:second",mesh_plan=plan).run(
        c.sim,c.rod,c.bindings,owner_resets=c.owners)
    before = (c.sim.current_time_step_index,c.rope.steps,len(c.calls))
    new = attach(c,plan,second)
    with pytest.raises(ValueError,match="clock|reset"):
        first.control(torch.zeros(1,24),observe=lambda row: None)
    assert before == (c.sim.current_time_step_index,c.rope.steps,len(c.calls))
    rows = []
    new.control(torch.zeros(1,24),observe=lambda row: rows.append(row))
    assert rows[0]["reaction_source_physics_step"] is None
    assert rows[0]["rod"]["reset_id"] == second["after_clock"]["reset_id"]
