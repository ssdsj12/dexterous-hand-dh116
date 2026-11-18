#!/usr/bin/env python3
"""Zero-CPU-copy pure-GPU Isaac Sim + Cosserat Rod (ropesim) pinch & lift runner."""
import json
import os
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

DT = 1.0 / 480.0
SUBSTEPS_PER_CTRL = 8


def scripted_action(step_idx: int, f_l: float, f_r: float, device):
    import torch

    a = torch.zeros((1, 24), dtype=torch.float32, device=device)
    if step_idx < 15:
        return a
    if step_idx < 75:
        if step_idx < 55:
            a[0, 17] = -0.8
        if f_l < 2.2:
            a[0, 1] = 0.22
            a[0, 2] = 0.26
        if f_r < 2.5:
            a[0, 7] = 0.22
            a[0, 8] = 0.28
        return a
    if step_idx < 95:
        return a
    if step_idx < 335:
        # Hold finger joint targets locked during lift; only micro-close if < 0.5 N
        if f_l < 0.5:
            a[0, 1] = 0.03
            a[0, 2] = 0.03
        if f_r < 0.5:
            a[0, 7] = 0.03
            a[0, 8] = 0.03
        a[0, 14] = 1.0
        a[0, 17] = 1.0
        return a
    return a


def main():
    from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as builder
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS, MIMIC, active_limits
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import run_owned_call
    from go2_pvcnn.control.dh116.tabletop_dual_hand_rod_runtime import PADS
    from scripts.dh116_pinch_startup_v3 import StartupTrace
    from scripts.verify_dh116_dual_pinch_interval import diagnostic_reset_owners
    from isaaclab.app import AppLauncher

    binary = Path(builder.BACKEND)
    sys.path.insert(0, str(binary.parents[1]))
    from ropesim import _ropesim  # noqa: F401

    app_launcher = AppLauncher(headless=True)
    app = app_launcher.app

    import isaaclab.sim as sim_utils
    import torch

    run_id = "gpu-" + uuid.uuid4().hex[:8]
    out_dir = ROOT / "runs/flexible_pinch_lift_demo" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=DT,
            device="cuda:0",
            render_interval=1,
            gravity=(0.0, 0.0, -9.81),
            physx=sim_utils.PhysxCfg(solver_type=0),
        )
    )
    ownership = StartupTrace(out_dir / "startup_trace.json")
    records = []

    def _run(owners):
        builder.build_reset_owned(
            sim,
            owners,
            project=ROOT,
            prepared=None,
            ownership=ownership,
            record=records.append,
            owner_resets=diagnostic_reset_owners(),
            run_uuid=run_id,
            episode_uid=run_id + ":ep0",
            physics_budget_end=10000,
            damping_override=0.0,
        )
        rod = owners.resources["runtime"]
        rope = rod.rope
        bindings = owners.resources["hand_bindings"]
        left_hand, right_hand = bindings[0].hand, bindings[1].hand
        left_mount, right_mount = bindings[0].mount, bindings[1].mount

        left_pad_ids = torch.tensor([left_hand.body_names.index(n) for n in PADS], device="cuda:0", dtype=torch.long)
        right_pad_ids = torch.tensor([right_hand.body_names.index(n) for n in PADS], device="cuda:0", dtype=torch.long)

        # Active & 11-joint coupling setup on GPU
        lo_np, hi_np = active_limits()
        q_lo = torch.tensor(lo_np, device="cuda:0", dtype=torch.float32)
        q_hi = torch.tensor(hi_np, device="cuda:0", dtype=torch.float32)

        def _hand_joint_setup(hand):
            act_ids = [hand.joint_names.index(n) for n in ACTIVE]
            all_ids = [hand.joint_names.index(n) for n in JOINTS]
            q_act = hand.data.joint_pos[0, act_ids].clone()
            return act_ids, all_ids, q_act

        l_act_ids, l_all_ids, l_q = _hand_joint_setup(left_hand)
        r_act_ids, r_all_ids, r_q = _hand_joint_setup(right_hand)
        mimic_map = {f: (ACTIVE.index(m), ratio, off) for f, m, ratio, off in MIMIC}
        active_idx = {n: i for i, n in enumerate(ACTIVE)}

        def _expand_11(q_act):
            out = torch.empty((1, 11), device="cuda:0", dtype=torch.float32)
            for j_idx, name in enumerate(JOINTS):
                if name in active_idx:
                    out[0, j_idx] = q_act[active_idx[name]]
                else:
                    m_i, ratio, off = mimic_map[name]
                    out[0, j_idx] = ratio * q_act[m_i] + off
            return out

        l_wrist_pos = left_hand.data.body_link_pos_w[:, left_mount.base_id].clone()
        l_wrist_quat = left_hand.data.body_link_quat_w[:, left_mount.base_id].clone()
        r_wrist_pos = right_hand.data.body_link_pos_w[:, right_mount.base_id].clone()
        r_wrist_quat = right_hand.data.body_link_quat_w[:, right_mount.base_id].clone()

        mesh_pos = torch.zeros((1, 11, 3), dtype=torch.float32, device="cuda:0")
        mesh_quat = torch.zeros((1, 11, 4), dtype=torch.float32, device="cuda:0")
        mesh_pos[0, 5] = torch.tensor([0.6, 0.0, 0.25], dtype=torch.float32, device="cuda:0")
        mesh_quat[0, 5, 3] = 1.0
        tele_false = torch.zeros((1,), dtype=torch.bool, device="cuda:0")

        rope_pos_view = torch.as_tensor(rope.positions())        # [1, 41, 3]
        wrench_view = torch.as_tensor(rope.mesh_wrenches_view()) # [1, 64, 6]

        init_pos = rope_pos_view[0].clone()
        init_med_z = float(torch.median(init_pos[:, 2]).item())
        init_min_z = float(torch.min(init_pos[:, 2]).item())

        max_med_lift = 0.0
        max_min_lift = 0.0
        hold_10mm_s = 0.0
        f_l = 0.0
        f_r = 0.0
        total_ctrl_steps = 390  # 6.5 s

        t0 = time.perf_counter()
        for step in range(total_ctrl_steps):
            act = scripted_action(step, f_l, f_r, device="cuda:0")
            for _sub in range(SUBSTEPS_PER_CTRL):
                # 1. Apply previous GPU rod wrenches directly to PhysX hands (COM shift on GPU)
                w = wrench_view[0, :11]
                for hand, pad_ids, sl in ((left_hand, left_pad_ids, slice(0, 5)), (right_hand, right_pad_ids, slice(6, 11))):
                    origin = hand.data.body_link_pos_w[0, pad_ids]
                    com = hand.data.body_com_pos_w[0, pad_ids]
                    f = w[sl, :3]
                    tau = w[sl, 3:] + torch.cross(origin - com, f, dim=-1)
                    hand.permanent_wrench_composer.reset()
                    hand.permanent_wrench_composer.set_forces_and_torques(
                        forces=f.unsqueeze(0), torques=tau.unsqueeze(0), body_ids=pad_ids.tolist(), is_global=True
                    )

                # 2. Pure-GPU 24D target integration (0.3 rad/s fingers, 0.01 m/s wrist)
                l_q = torch.clamp(l_q + act[0, 0:6] * (0.3 * DT), q_lo, q_hi)
                r_q = torch.clamp(r_q + act[0, 6:12] * (0.3 * DT), q_lo, q_hi)
                l_wrist_pos = l_wrist_pos + act[0, 12:15].unsqueeze(0) * (0.01 * DT)
                r_wrist_pos = r_wrist_pos + act[0, 15:18].unsqueeze(0) * (0.01 * DT)

                left_hand.set_joint_position_target(_expand_11(l_q), joint_ids=l_all_ids)
                right_hand.set_joint_position_target(_expand_11(r_q), joint_ids=r_all_ids)
                left_mount.move(torch.cat((l_wrist_pos, l_wrist_quat), dim=-1))
                right_mount.move(torch.cat((r_wrist_pos, r_wrist_quat), dim=-1))

                left_hand.write_data_to_sim()
                right_hand.write_data_to_sim()
                sim.step(render=False)
                left_hand.update(DT)
                right_hand.update(DT)

                # 3. Updated pad poses -> GPU buffer -> ropesim
                mesh_pos[0, 0:5] = left_hand.data.body_link_pos_w[0, left_pad_ids]
                lq = left_hand.data.body_link_quat_w[0, left_pad_ids]
                mesh_quat[0, 0:5] = lq[:, [1, 2, 3, 0]]
                mesh_pos[0, 6:11] = right_hand.data.body_link_pos_w[0, right_pad_ids]
                rq = right_hand.data.body_link_quat_w[0, right_pad_ids]
                mesh_quat[0, 6:11] = rq[:, [1, 2, 3, 0]]

                rope.set_mesh_poses_all(mesh_pos, mesh_quat, tele_false)
                rope.step_free(DT, 12)
                if rope.last_error:
                    raise RuntimeError(f"ropesim error: {rope.last_error}")

            z = rope_pos_view[0, :, 2]
            med_lift = float(torch.median(z).item()) - init_med_z
            min_lift = float(torch.min(z).item()) - init_min_z
            f_l = float(torch.linalg.norm(wrench_view[0, 0:5, :3], dim=-1).sum().item())
            f_r = float(torch.linalg.norm(wrench_view[0, 6:11, :3], dim=-1).sum().item())
            max_med_lift = max(max_med_lift, med_lift)
            max_min_lift = max(max_min_lift, min_lift)
            if 0.5 * (med_lift + min_lift) >= 0.01:
                hold_10mm_s += 1.0 / 60.0

            if step in (0, 60, 90, 95, 100, 105, 110, 120, 150):
                lf_pads = torch.linalg.norm(wrench_view[0, 0:2, :3], dim=-1).tolist()
                rf_pads = torch.linalg.norm(wrench_view[0, 6:8, :3], dim=-1).tolist()
                lp = left_hand.data.body_link_pos_w[0, left_pad_ids[:2]].cpu().numpy()
                rp = right_hand.data.body_link_pos_w[0, right_pad_ids[:2]].cpu().numpy()
                print(
                    f"[DBG step={step:3d} t={step/60:.2f}s] "
                    f"L_q={l_q[:3].cpu().numpy().round(3).tolist()} R_q={r_q[:3].cpu().numpy().round(3).tolist()} "
                    f"L_pads={[round(x,2) for x in lf_pads]} R_pads={[round(x,2) for x in rf_pads]} "
                    f"L_z={lp[:,2].round(4).tolist()} R_z={rp[:,2].round(4).tolist()}",
                    flush=True,
                )
            if step % 30 == 0 or step == total_ctrl_steps - 1:
                elapsed = time.perf_counter() - t0
                print(
                    f"[step {step:3d} t={step/60:.2f}s wall={elapsed:.1f}s] "
                    f"med={med_lift*1000:+.2f}mm min={min_lift*1000:+.2f}mm "
                    f"hold10={hold_10mm_s:.2f}s F_L={f_l:.2f}N F_R={f_r:.2f}N",
                    flush=True,
                )

        elapsed = time.perf_counter() - t0
        summary = {
            "wall_seconds": round(elapsed, 2),
            "sim_seconds": total_ctrl_steps / 60.0,
            "steps_per_sec": round((total_ctrl_steps * SUBSTEPS_PER_CTRL) / elapsed, 1),
            "max_median_lift_mm": round(max_med_lift * 1000, 2),
            "max_min_lift_mm": round(max_min_lift * 1000, 2),
            "hold_10mm_s": round(hold_10mm_s, 3),
            "curriculum_stage1_passed": hold_10mm_s >= 0.5,
        }
        (out_dir / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("SUMMARY:", json.dumps(summary), flush=True)

    run_owned_call(_run, synchronize=lambda: None, ownership=ownership)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
