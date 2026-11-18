#!/usr/bin/env python3
"""Single-env long-horizon (>10s capable) scripted flexible-wire pinch & lift demo."""
import json
import os
from pathlib import Path
import sys
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def scripted_action(step_idx: int):
    import torch

    a = torch.zeros((1, 24), dtype=torch.float64)
    if step_idx < 15:
        # Settle wire on table
        return a
    if step_idx < 75:
        # Close thumb (finger12 -> idx 1, 7) and index (finger21 -> idx 2, 8)
        a[0, 1] = 0.25
        a[0, 2] = 0.30
        a[0, 7] = 0.25
        a[0, 8] = 0.30
        return a
    if step_idx < 105:
        # Prelift hold
        a[0, 1] = 0.05
        a[0, 2] = 0.05
        a[0, 7] = 0.05
        a[0, 8] = 0.05
        return a
    if step_idx < 285:
        # Lift +Z at 0.01 m/s for 3.0s (~30mm) with gentle squeeze
        a[0, 1] = 0.04
        a[0, 2] = 0.04
        a[0, 7] = 0.04
        a[0, 8] = 0.04
        a[0, 14] = 1.0
        a[0, 17] = 1.0
        return a
    # Hold at lifted height
    a[0, 1] = 0.02
    a[0, 2] = 0.02
    a[0, 7] = 0.02
    a[0, 8] = 0.02
    return a


def main():
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import run_owned_call
    from scripts.dh116_pinch_startup_v3 import StartupTrace
    from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as builder
    from scripts.verify_dh116_dual_pinch_interval import diagnostic_reset_owners
    from isaaclab.app import AppLauncher

    binary = Path(builder.BACKEND)
    sys.path.insert(0, str(binary.parents[1]))
    from ropesim import _ropesim  # noqa: F401

    app_launcher = AppLauncher(headless=True)
    app = app_launcher.app

    import isaaclab.sim as sim_utils

    run_id = "demo-" + uuid.uuid4().hex[:8]
    out_dir = ROOT / "runs/flexible_pinch_lift_demo" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=1 / 480,
            device="cuda:0",
            render_interval=1,
            gravity=(0.0, 0.0, -9.81),
            physx=sim_utils.PhysxCfg(solver_type=0),
        )
    )

    records = []
    ownership = StartupTrace(out_dir / 'startup_trace.json')
    total_control_steps = 390  # 6.5 s = 3120 physics steps

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
            physics_budget_end=6000,
            damping_override=0.0,
        )
        ep = owners.resources["dual_episode"]
        rod = owners.resources["runtime"]

        import torch
        init_nodes = np.asarray(rod._base.nodes)
        init_med_z = float(np.median(init_nodes[:, 2]))
        init_min_z = float(np.min(init_nodes[:, 2]))

        max_med_lift = 0.0
        max_min_lift = 0.0
        hold_stage1_s = 0.0
        term_reason = None

        for step in range(total_control_steps):
            act = scripted_action(step)
            res = ep.control(act)
            nodes = torch.as_tensor(rod.rope.positions()).cpu().numpy()[0]
            med_lift = float(np.median(nodes[:, 2])) - init_med_z
            min_lift = float(np.min(nodes[:, 2])) - init_min_z
            max_med_lift = max(max_med_lift, med_lift)
            max_min_lift = max(max_min_lift, min_lift)
            if 0.5 * (med_lift + min_lift) >= 0.01:
                hold_stage1_s += 1.0 / 60.0
            if step % 60 == 0 or step == total_control_steps - 1 or res.get("termination_reason"):
                w = (rod._base.last or {}).get("named_wrenches") or {}
                lf = sum(float(np.linalg.norm(w[k][:3])) for k in w if k.startswith("left."))
                rf = sum(float(np.linalg.norm(w[k][:3])) for k in w if k.startswith("right."))
                print(
                    f"[step {step:3d} t={step/60:.2f}s] med_lift={med_lift*1000:+.2f}mm "
                    f"min_lift={min_lift*1000:+.2f}mm hold_10mm_s={hold_stage1_s:.2f}s "
                    f"F_L={lf:.3f}N F_R={rf:.3f}N term={res.get('termination_reason')}",
                    flush=True,
                )
            if res.get("termination_reason"):
                term_reason = res["termination_reason"]
                break

        summary = {
            "total_control_steps": step + 1,
            "physics_steps": rod.clock()["physics_step"],
            "init_median_z_m": init_med_z,
            "init_min_z_m": init_min_z,
            "max_median_lift_m": max_med_lift,
            "max_min_lift_m": max_min_lift,
            "hold_10mm_s": hold_stage1_s,
            "curriculum_stage1_passed": hold_stage1_s >= 0.5,
            "termination_reason": term_reason,
        }
        (out_dir / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("SUMMARY:", json.dumps(summary))

    run_owned_call(_run, synchronize=lambda: None, ownership=ownership)
    app.close(wait_for_replicator=False)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
