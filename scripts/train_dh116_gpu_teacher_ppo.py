#!/usr/bin/env python3
"""16-env pure-GPU zero-copy Teacher BC warm-start + PPO training on Isaac Sim + Cosserat Rod."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

DT = 1.0 / 480.0
SUBSTEPS_PER_CTRL = 8


def scripted_batch_action(step_idx: int, f_l_pads, f_r_pads, mode: str, device):
    import torch

    N = f_l_pads.shape[0]
    a = torch.zeros((N, 24), dtype=torch.float32, device=device)
    if step_idx < 15:
        return a

    f_l = f_l_pads.sum(dim=-1)
    f_r = f_r_pads.sum(dim=-1)

    if mode == "pinch":
        if step_idx < 75:
            if step_idx < 55:
                a[:, 14] = -0.6   # seat left hand 4mm deeper on wire to avoid fingertip-edge slip
                a[:, 17] = -0.8
            mask_l = f_l < 2.4
            mask_r = f_r < 2.8
            a[mask_l, 1] = 0.22
            a[mask_l, 2] = 0.26
            a[mask_r, 7] = 0.22
            a[mask_r, 8] = 0.28
            return a
        if step_idx < 95:
            return a
        # Emergency only below 1.0 N; otherwise lock pinch pose to avoid creeping past wire saddle
        mask_l = f_l < 1.0
        mask_r = f_r < 1.0
        a[mask_l, 1] = 0.02
        a[mask_l, 2] = 0.02
        a[mask_r, 7] = 0.02
        a[mask_r, 8] = 0.02
        if step_idx < 375:
            a[:, 14] = 1.0
            a[:, 17] = 1.0
        return a

    if mode == "tripod":
        if step_idx < 75:
            if step_idx < 55:
                a[:, 14] = -0.60
                a[:, 17] = -0.60
            ml = f_l < 1.8
            a[ml & (f_l_pads[:, 0] < 1.0), 1] = 0.26
            a[ml & (f_l_pads[:, 1] < 0.65), 2] = 0.22
            a[ml & (f_l_pads[:, 2] < 0.65), 3] = 0.22
            mr = f_r < 2.2
            a[mr & (f_r_pads[:, 0] < 1.2), 7] = 0.26
            a[mr & (f_r_pads[:, 1] < 0.70), 8] = 0.22
            a[mr & (f_r_pads[:, 2] < 0.70), 9] = 0.22
            return a
        if step_idx < 95:
            return a
        ml = f_l < 1.0
        mr = f_r < 1.0
        a[ml, 1] = 0.02; a[ml, 2] = 0.02; a[ml, 3] = 0.02
        a[mr, 7] = 0.02; a[mr, 8] = 0.02; a[mr, 9] = 0.02
        if step_idx < 375:
            a[:, 14] = 1.0
            a[:, 17] = 1.0
        return a

    # mode == "power"
    if step_idx < 75:
        if step_idx < 55:
            a[:, 14] = -0.50
            a[:, 17] = -0.50
        ml = f_l < 2.3
        mr = f_r < 2.3
        a[ml & (f_l_pads[:, 0] < 1.1), 1] = 0.26
        a[mr & (f_r_pads[:, 0] < 1.1), 7] = 0.26
        for ch, pidx in ((2, 1), (3, 2), (4, 3), (5, 4)):
            a[(f_l_pads[:, pidx] < 0.05) | (ml & (f_l_pads[:, pidx] < 0.65)), ch] = 0.22
            a[(f_r_pads[:, pidx] < 0.05) | (mr & (f_r_pads[:, pidx] < 0.65)), 6 + ch] = 0.22
        return a
    if step_idx < 95:
        return a
    ml = f_l < 1.0
    mr = f_r < 1.0
    for ch in (1, 2, 3, 4, 5):
        a[ml, ch] = 0.02
        a[mr, 6 + ch] = 0.02
    if step_idx < 375:
        a[:, 14] = 1.0
        a[:, 17] = 1.0
    return a


def build_actor_critic(obs_dim: int, act_dim: int, init_std: float, device):
    import torch
    import torch.nn as nn

    class TeacherAC(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = nn.Sequential(
                nn.Linear(obs_dim, 256),
                nn.ELU(),
                nn.Linear(256, 256),
                nn.ELU(),
                nn.Linear(256, act_dim),
            )
            self.critic = nn.Sequential(
                nn.Linear(obs_dim, 256),
                nn.ELU(),
                nn.Linear(256, 256),
                nn.ELU(),
                nn.Linear(256, 1),
            )
            self.log_std = nn.Parameter(torch.full((act_dim,), float(np.log(init_std))))

        def forward(self, obs):
            mean = (1.05 * torch.tanh(self.actor(obs))).clamp(-1.0, 1.0)
            std = torch.exp(self.log_std).clamp(0.004, 0.06)
            val = self.critic(obs).squeeze(-1)
            return mean, std, val

    return TeacherAC().to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="pinch", choices=["pinch", "tripod", "power"])
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=480, help="control steps per rollout (480 = 8.0s)")
    parser.add_argument("--updates", type=int, default=30)
    parser.add_argument("--bc-epochs", type=int, default=120)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True)
    app = app_launcher.app

    from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as builder
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_scene_preparation as scene_api
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS, MIMIC, active_limits, expand
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import box_sdf
    from go2_pvcnn.control.dh116.tabletop_dual_hand_rod_runtime import PADS

    binary = Path(builder.BACKEND)
    sys.path.insert(0, str(binary.parents[1]))
    from ropesim import RopeBatch, sdf

    from go2_pvcnn.control.dh116.tabletop_palm_mount import (
        ANGULAR_JOINT_NAMES,
        CONTRACT as WRIST,
        JOINT_NAMES,
        ROTATION_CONTRACT,
        local_translation,
    )
    from go2_pvcnn.control.dh116.tabletop_vec_isaac import spawn_scene
    from go2_pvcnn.control.dh116.tabletop_velocity_limits import ACTIVE_VELOCITY_LIMIT_RAD_S, passive_velocity_limit

    import isaaclab.sim as sim_utils
    import torch
    import torch.nn as nn
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from pxr import Usd, UsdGeom, UsdPhysics

    device = "cuda:0"
    N = args.num_envs
    run_id = f"ppo{N}-{args.mode}-{uuid.uuid4().hex[:8]}"
    out_dir = ROOT / "runs/gpu_teacher_ppo" / run_id
    if not args.eval_only:
        out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INIT] mode={args.mode}, num_envs={N}, horizon={args.horizon} ({args.horizon/60:.2f}s), updates={args.updates}, out={out_dir}", flush=True)

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=DT,
            device=device,
            render_interval=1,
            gravity=(0.0, 0.0, -9.81),
            physx=sim_utils.PhysxCfg(solver_type=0),
        )
    )

    prepared = scene_api.prepare_scene(ROOT)
    sp_args, layout, sources = scene_api.spawn_parameters(ROOT, prepared)
    UsdGeom.Xform.Define(sim.stage, "/World/envs")
    env_origins_np = np.array(
        [[float((i % 4) - 1.5) * 3.0, float((i // 4) - 1.5) * 3.0, 0.0] for i in range(N)],
        dtype=np.float32,
    ) if N > 4 else np.array([[0.0, float(i) * 3.0, 0.0] for i in range(N)], dtype=np.float32)
    env_origins = torch.tensor(env_origins_np, dtype=torch.float32, device=device)

    for i in range(N):
        prefix = f"/World/envs/env_{i}"
        spawn_scene(sim, sp_args, layout, sources, prefix, env_origins_np[i], external_wire=True)
        for p in Usd.PrimRange(sim.stage.GetPrimAtPath(f"{prefix}/Table")):
            if p.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Set(False)

    def _make_art(side):
        actuators = {
            "active": ImplicitActuatorCfg(
                joint_names_expr=list(ACTIVE), stiffness=10.0, damping=0.2,
                effort_limit_sim=0.15, velocity_limit_sim=ACTIVE_VELOCITY_LIMIT_RAD_S,
            ),
            "passive": ImplicitActuatorCfg(
                joint_names_expr=[n for n in JOINTS if n not in ACTIVE], stiffness=0.0, damping=0.0,
                effort_limit_sim=0.0, velocity_limit_sim=passive_velocity_limit("adaptive_grasp", 3.0),
            ),
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=list(JOINT_NAMES), stiffness=WRIST["stiffness_n_m"],
                damping=WRIST["damping_n_s_m"], effort_limit_sim=WRIST["effort_limit_n"],
                velocity_limit_sim=WRIST["velocity_limit_m_s"],
            ),
            "wrist_rotation": ImplicitActuatorCfg(
                joint_names_expr=list(ANGULAR_JOINT_NAMES), stiffness=ROTATION_CONTRACT["stiffness_nm_rad"],
                damping=ROTATION_CONTRACT["damping_nms_rad"], effort_limit_sim=ROTATION_CONTRACT["effort_limit_nm"],
                velocity_limit_sim=ROTATION_CONTRACT["velocity_limit_rad_s"],
            ),
        }
        init_j = dict(zip(JOINTS, map(float, expand(layout["active_joints"][side]))))
        return Articulation(
            ArticulationCfg(
                prim_path=f"/World/envs/env_.*/{side}Hand",
                spawn=None,
                init_state=ArticulationCfg.InitialStateCfg(joint_pos=init_j),
                actuators=actuators,
            )
        )

    left_hands = _make_art("left")
    right_hands = _make_art("right")
    sim.reset()
    left_hands.update(DT)
    right_hands.update(DT)

    # Load reset seed & rod seed for selected mode
    cfg_rod = json.loads((ROOT / "configs/dh116_tasks/flexible_pinch_undamped_v1.json").read_text())
    r_cfg = cfg_rod["rod"]
    if args.mode == "pinch":
        pair_packet = builder.load_pair_proposal()
        seed = pair_packet["reset_seed"]
        base_nodes = np.asarray(pair_packet["rod_seed"]["proposed_nodes_m"], dtype=np.float32)  # [41, 3]
        seed_joint_11 = [seed["joint_positions"][0], seed["joint_positions"][1]]
        seed_root_list = seed["reset_root_pose_w"]
        req_pad_cnt = 2
    else:
        style_cfg = json.loads((ROOT / "configs/dh116_training/gpu_teacher_style_seeds_v1.json").read_text())
        s_entry = style_cfg["styles"][args.mode]
        rn = style_cfg["rod_seed_nodes"]
        y_vals = np.linspace(float(rn["y_min_m"]), float(rn["y_max_m"]), int(rn["num_nodes"]), dtype=np.float32)
        base_nodes = np.zeros((int(rn["num_nodes"]), 3), dtype=np.float32)
        base_nodes[:, 0] = float(rn["x_m"])
        base_nodes[:, 1] = y_vals
        base_nodes[:, 2] = float(rn["z_m"])
        seed_joint_11 = [
            expand(s_entry["left"]["active_q_rad"]).tolist(),
            expand(s_entry["right"]["active_q_rad"]).tolist(),
        ]
        seed_root_list = [s_entry["left"]["base_pose_wxyz"], s_entry["right"]["base_pose_wxyz"]]
        req_pad_cnt = 3 if args.mode == "tripod" else 5

    nodes_batch = base_nodes[None, :, :] + env_origins_np[:, None, :]  # [N, 41, 3]

    rope = RopeBatch.from_positions(
        nodes_batch,
        density=r_cfg["linear_density_kg_m"],
        k_stretch=r_cfg["axial_stiffness_n"],
        k_bend=r_cfg["bend_stiffness_n_m2"],
        max_h=DT / r_cfg["solver_substeps"],
        num_itr=r_cfg["solver_iterations"],
        pinned=np.zeros((N, len(base_nodes)), dtype=bool),
        fix_root_orientation=False,
        straight_rest=True,
        deterministic=True,
    )
    rope.set_gravity(np.asarray(r_cfg["gravity_world_m_s2"], dtype=np.float32))
    rope.set_contact(radius=r_cfg["radius_m"], mu_s=r_cfg["mu_s"], mu_k=r_cfg["mu_k"], v_slip=r_cfg["v_slip_m_s"], kappa=r_cfg["kappa"])
    rope.set_rope_contact(False)
    rope.damping = float(r_cfg["damping"])
    rope.drag = r_cfg["drag"]

    sdf_arrays = scene_api.load_sdf_arrays(prepared)
    grids = {k: sdf.SdfGrid(**v) for k, v in sdf_arrays.items()}

    l_pad_ids = [left_hands.body_names.index(n) for n in PADS]
    r_pad_ids = [right_hands.body_names.index(n) for n in PADS]
    l_base_id = left_hands.body_names.index("base_link")
    r_base_id = right_hands.body_names.index("base_link")

    l_wrist_ids = [left_hands.joint_names.index(n) for n in JOINT_NAMES + ANGULAR_JOINT_NAMES]
    r_wrist_ids = [right_hands.joint_names.index(n) for n in JOINT_NAMES + ANGULAR_JOINT_NAMES]
    l_act_ids = [left_hands.joint_names.index(n) for n in ACTIVE]
    r_act_ids = [right_hands.joint_names.index(n) for n in ACTIVE]
    l_all_ids = [left_hands.joint_names.index(n) for n in JOINTS]
    r_all_ids = [right_hands.joint_names.index(n) for n in JOINTS]

    # Seed joint targets [2, 17]
    seed_q = torch.zeros((2, 17), device=device, dtype=torch.float32)
    for hi, h in enumerate((left_hands, right_hands)):
        for j, name in enumerate(JOINTS):
            seed_q[hi, h.joint_names.index(name)] = float(seed_joint_11[hi][j])
    seed_root_base = torch.tensor(seed_root_list, device=device, dtype=torch.float32)  # [2, 7]

    from go2_pvcnn.control.dh116.tabletop_material import hand_material_contract
    hand_mat = hand_material_contract(cfg_rod["hand"]["material_profile"])["hand_contact_material"]
    pad_mu_s = max(r_cfg["mu_s"], hand_mat["static_friction"])
    pad_mu_k = max(r_cfg["mu_k"], hand_mat["dynamic_friction"])

    all_env_ids_np = np.arange(N, dtype=np.int32)
    for hi, (h, r_pose) in enumerate(((left_hands, seed_root_base[0]), (right_hands, seed_root_base[1]))):
        root_w = r_pose.unsqueeze(0).repeat(N, 1)
        root_w[:, :3] += env_origins
        vel_0 = torch.zeros((N, 6), device=device, dtype=torch.float32)
        q_n = seed_q[hi : hi + 1].repeat(N, 1)
        qd_0 = torch.zeros_like(q_n)
        h.write_root_pose_to_sim(root_w)
        h.write_root_velocity_to_sim(vel_0)
        h.write_joint_state_to_sim(q_n, qd_0)
        h.set_joint_position_target(q_n)
        h.write_data_to_sim()
        h.update(DT)

    for i, name in enumerate(PADS):
        pos = left_hands.data.body_link_pos_w[:, l_pad_ids[i]].cpu().numpy()
        quat = left_hands.data.body_link_quat_w[:, l_pad_ids[i]][:, [1, 2, 3, 0]].cpu().numpy()
        rope.add_mesh(*grids["left." + name].as_args(), pos=pos, quat=quat, mu_s=pad_mu_s, mu_k=pad_mu_k)

    table_size = np.asarray(prepared["table"]["size_m"], dtype=np.float32)
    table_center = np.asarray(prepared["table"]["center_m"], dtype=np.float32)
    half = table_size / 2
    tbl_cfg = cfg_rod["table"]
    tbl_grid = sdf.from_function(lambda p: box_sdf(p, table_size), -half - tbl_cfg["sdf_margin_m"], half + tbl_cfg["sdf_margin_m"], tbl_cfg["sdf_spacing_m"])
    tbl_pos = table_center[None, :] + env_origins_np
    tbl_quat = np.zeros((N, 4), dtype=np.float32); tbl_quat[:, 3] = 1.0
    rope.add_mesh(*tbl_grid.as_args(), pos=tbl_pos, quat=tbl_quat, mu_s=tbl_cfg["mu_s"], mu_k=tbl_cfg["mu_k"])

    for i, name in enumerate(PADS):
        pos = right_hands.data.body_link_pos_w[:, r_pad_ids[i]].cpu().numpy()
        quat = right_hands.data.body_link_quat_w[:, r_pad_ids[i]][:, [1, 2, 3, 0]].cpu().numpy()
        rope.add_mesh(*grids["right." + name].as_args(), pos=pos, quat=quat, mu_s=pad_mu_s, mu_k=pad_mu_k)

    rope.enable_contact_forces(True)

    rope_pos_view = torch.as_tensor(rope.positions())          # [N, 41, 3]
    wrench_view = torch.as_tensor(rope.mesh_wrenches_view())   # [N, 64, 6]
    mesh_pos = torch.zeros((N, 11, 3), dtype=torch.float32, device=device)
    mesh_quat = torch.zeros((N, 11, 4), dtype=torch.float32, device=device)
    mesh_pos[:, 5] = torch.tensor(tbl_pos, dtype=torch.float32, device=device)
    mesh_quat[:, 5, 3] = 1.0
    tele_buf = torch.zeros((N,), dtype=torch.bool, device=device)

    lo_np, hi_np = active_limits()
    q_lo = torch.tensor(lo_np, device=device, dtype=torch.float32)
    q_hi = torch.tensor(hi_np, device=device, dtype=torch.float32)
    mimic_map = {f: (ACTIVE.index(m), ratio, off) for f, m, ratio, off in MIMIC}
    active_idx = {n: i for i, n in enumerate(ACTIVE)}

    def _expand_11(q_act):
        cols = []
        for name in JOINTS:
            if name in active_idx:
                cols.append(q_act[:, active_idx[name]])
            else:
                m_i, ratio, off = mimic_map[name]
                cols.append(ratio * q_act[:, m_i] + off)
        return torch.stack(cols, dim=-1)

    l_q = torch.zeros((N, 6), device=device, dtype=torch.float32)
    r_q = torch.zeros((N, 6), device=device, dtype=torch.float32)
    l_wrist_pos = torch.zeros((N, 3), device=device, dtype=torch.float32)
    r_wrist_pos = torch.zeros((N, 3), device=device, dtype=torch.float32)
    l_wrist_ang = torch.zeros((N, 3), device=device, dtype=torch.float32)
    r_wrist_ang = torch.zeros((N, 3), device=device, dtype=torch.float32)
    prev_act = torch.zeros((N, 24), device=device, dtype=torch.float32)
    hold10_s = torch.zeros((N,), device=device, dtype=torch.float32)
    hold30_s = torch.zeros((N,), device=device, dtype=torch.float32)
    stage1_awarded = torch.zeros((N,), device=device, dtype=torch.bool)
    stage2_awarded = torch.zeros((N,), device=device, dtype=torch.bool)
    step_in_ep = torch.zeros((N,), device=device, dtype=torch.long)

    init_med_z = float(np.median(base_nodes[:, 2]))
    init_min_z = float(np.min(base_nodes[:, 2]))
    wire_ref_center = torch.tensor([0.43, 0.0, 0.308], device=device, dtype=torch.float32)
    wire_nodes_idx = torch.tensor([0, 10, 20, 30, 40], device=device, dtype=torch.long)
    l_pinch_node = 36
    r_pinch_node = 4

    def reset_all():
        nonlocal l_q, r_q, l_wrist_pos, r_wrist_pos, l_wrist_ang, r_wrist_ang, prev_act, hold10_s, hold30_s, stage1_awarded, stage2_awarded, step_in_ep
        for hi, (h, r_pose) in enumerate(((left_hands, seed_root_base[0]), (right_hands, seed_root_base[1]))):
            root_w = r_pose.unsqueeze(0).repeat(N, 1)
            root_w[:, :3] += env_origins
            vel_0 = torch.zeros((N, 6), device=device, dtype=torch.float32)
            q_n = seed_q[hi : hi + 1].repeat(N, 1)
            qd_0 = torch.zeros_like(q_n)
            h.permanent_wrench_composer.reset()
            h.write_root_pose_to_sim(root_w)
            h.write_root_velocity_to_sim(vel_0)
            h.write_joint_state_to_sim(q_n, qd_0)
            h.set_joint_position_target(q_n)
            h.write_data_to_sim()
            h.update(DT)

        rope.reset_envs(all_env_ids_np)
        mesh_pos[:, 0:5] = left_hands.data.body_link_pos_w[:, l_pad_ids]
        mesh_quat[:, 0:5] = left_hands.data.body_link_quat_w[:, l_pad_ids][:, :, [1, 2, 3, 0]]
        mesh_pos[:, 6:11] = right_hands.data.body_link_pos_w[:, r_pad_ids]
        mesh_quat[:, 6:11] = right_hands.data.body_link_quat_w[:, r_pad_ids][:, :, [1, 2, 3, 0]]
        tele_buf.fill_(True)
        rope.set_mesh_poses_all(mesh_pos, mesh_quat, tele_buf)

        l_q = left_hands.data.joint_pos[:, l_act_ids].clone()
        r_q = right_hands.data.joint_pos[:, r_act_ids].clone()
        l_wrist_pos = left_hands.data.body_link_pos_w[:, l_base_id].clone()
        r_wrist_pos = right_hands.data.body_link_pos_w[:, r_base_id].clone()
        l_wrist_ang.zero_()
        r_wrist_ang.zero_()
        prev_act.zero_()
        hold10_s.zero_()
        hold30_s.zero_()
        stage1_awarded.zero_()
        stage2_awarded.zero_()
        step_in_ep.zero_()

    def get_obs():
        l_qd = left_hands.data.joint_vel[:, l_act_ids]
        r_qd = right_hands.data.joint_vel[:, r_act_ids]
        l_base_pos = left_hands.data.body_link_pos_w[:, l_base_id] - env_origins - wire_ref_center
        r_base_pos = right_hands.data.body_link_pos_w[:, r_base_id] - env_origins - wire_ref_center
        l_base_quat = left_hands.data.body_link_quat_w[:, l_base_id]
        r_base_quat = right_hands.data.body_link_quat_w[:, r_base_id]
        pad_f = torch.cat((
            torch.linalg.norm(wrench_view[:, 0:5, :3], dim=-1),
            torch.linalg.norm(wrench_view[:, 6:11, :3], dim=-1),
        ), dim=-1).clamp(0.0, 20.0) / 5.0
        w_nodes = (rope_pos_view[:, wire_nodes_idx, :] - env_origins.unsqueeze(1) - wire_ref_center).reshape(N, 15)
        z = rope_pos_view[:, :, 2]
        med_lift = torch.median(z, dim=-1).values - init_med_z
        min_lift = torch.min(z, dim=-1).values - init_min_z
        z_span = torch.max(z, dim=-1).values - torch.min(z, dim=-1).values
        wire_stats = torch.stack((med_lift * 20.0, min_lift * 20.0, z_span * 20.0), dim=-1)
        t_norm = (step_in_ep.float() / float(args.horizon)).unsqueeze(-1)
        return torch.cat((
            l_q, r_q, l_qd * 0.5, r_qd * 0.5,
            l_base_pos * 5.0, r_base_pos * 5.0, l_base_quat, r_base_quat,
            pad_f, w_nodes * 5.0, wire_stats, t_norm,
        ), dim=-1)  # 67D

    def step_env(actions):
        nonlocal l_q, r_q, l_wrist_pos, r_wrist_pos, l_wrist_ang, r_wrist_ang, prev_act
        act = actions.clamp(-1.0, 1.0)
        for _sub in range(SUBSTEPS_PER_CTRL):
            for h, pad_ids, sl in ((left_hands, l_pad_ids, slice(0, 5)), (right_hands, r_pad_ids, slice(6, 11))):
                origin = h.data.body_link_pos_w[:, pad_ids]
                com = h.data.body_com_pos_w[:, pad_ids]
                f = wrench_view[:, sl, :3]
                tau = wrench_view[:, sl, 3:] + torch.cross(origin - com, f, dim=-1)
                h.permanent_wrench_composer.reset()
                h.permanent_wrench_composer.set_forces_and_torques(forces=f, torques=tau, body_ids=pad_ids, is_global=True)

            l_q = torch.clamp(l_q + act[:, 0:6] * (0.3 * DT), q_lo, q_hi)
            r_q = torch.clamp(r_q + act[:, 6:12] * (0.3 * DT), q_lo, q_hi)
            l_wrist_pos = l_wrist_pos + act[:, 12:15] * (0.01 * DT)
            r_wrist_pos = r_wrist_pos + act[:, 15:18] * (0.01 * DT)
            l_wrist_ang = torch.clamp(l_wrist_ang + act[:, 18:21] * (0.5 * DT), -1.5, 1.5)
            r_wrist_ang = torch.clamp(r_wrist_ang + act[:, 21:24] * (0.5 * DT), -1.5, 1.5)

            left_hands.set_joint_position_target(_expand_11(l_q), joint_ids=l_all_ids)
            right_hands.set_joint_position_target(_expand_11(r_q), joint_ids=r_all_ids)
            l_loc = torch.cat((local_translation(l_wrist_pos, left_hands.data.root_pose_w), l_wrist_ang), dim=-1)
            r_loc = torch.cat((local_translation(r_wrist_pos, right_hands.data.root_pose_w), r_wrist_ang), dim=-1)
            left_hands.set_joint_position_target(l_loc, joint_ids=l_wrist_ids)
            right_hands.set_joint_position_target(r_loc, joint_ids=r_wrist_ids)

            left_hands.write_data_to_sim()
            right_hands.write_data_to_sim()
            sim.step(render=False)
            left_hands.update(DT)
            right_hands.update(DT)

            mesh_pos[:, 0:5] = left_hands.data.body_link_pos_w[:, l_pad_ids]
            mesh_quat[:, 0:5] = left_hands.data.body_link_quat_w[:, l_pad_ids][:, :, [1, 2, 3, 0]]
            mesh_pos[:, 6:11] = right_hands.data.body_link_pos_w[:, r_pad_ids]
            mesh_quat[:, 6:11] = right_hands.data.body_link_quat_w[:, r_pad_ids][:, :, [1, 2, 3, 0]]
            tele_buf.zero_()
            rope.set_mesh_poses_all(mesh_pos, mesh_quat, tele_buf)
            rope.step_free(DT, 12)
            if rope.last_error:
                raise RuntimeError(f"ropesim error: {rope.last_error}")

        z = rope_pos_view[:, :, 2]
        med_lift = torch.median(z, dim=-1).values - init_med_z
        min_lift = torch.min(z, dim=-1).values - init_min_z
        eff_lift = 0.7 * min_lift + 0.3 * med_lift

        f_l_pads = torch.linalg.norm(wrench_view[:, 0:5, :3], dim=-1)
        f_r_pads = torch.linalg.norm(wrench_view[:, 6:11, :3], dim=-1)
        f_l = f_l_pads.sum(dim=-1)
        f_r = f_r_pads.sum(dim=-1)

        l_pad_mean = left_hands.data.body_link_pos_w[:, l_pad_ids[:req_pad_cnt]].mean(dim=1)
        r_pad_mean = right_hands.data.body_link_pos_w[:, r_pad_ids[:req_pad_cnt]].mean(dim=1)
        d_l = torch.linalg.norm(l_pad_mean - rope_pos_view[:, l_pinch_node], dim=-1)
        d_r = torch.linalg.norm(r_pad_mean - rope_pos_view[:, r_pinch_node], dim=-1)
        r_dist = 0.2 * torch.exp(-15.0 * (d_l + d_r))
        r_contact = 0.4 * (torch.clamp(f_l / 2.0, 0.0, 1.0) + torch.clamp(f_r / 2.0, 0.0, 1.0))
        if req_pad_cnt > 2:
            active_l = (f_l_pads[:, :req_pad_cnt] > 0.05).float().mean(dim=-1)
            active_r = (f_r_pads[:, :req_pad_cnt] > 0.05).float().mean(dim=-1)
            r_contact = r_contact + 0.2 * (active_l + active_r)
        has_grip = (f_l > 0.3) & (f_r > 0.3)
        r_lift = torch.where(
            has_grip,
            4.0 * torch.clamp(eff_lift / 0.03, 0.0, 1.0) + 2.5 * torch.clamp((min_lift - 0.02) / 0.025, 0.0, 1.0),
            0.0 * eff_lift,
        )
        bilat10 = has_grip & (min_lift >= 0.01)
        bilat30 = has_grip & (min_lift >= 0.03)
        hold10_s[:] = torch.where(bilat10, hold10_s + 1.0 / 60.0, torch.clamp(hold10_s - 2.0 / 60.0, min=0.0))
        hold30_s[:] = torch.where(bilat30, hold30_s + 1.0 / 60.0, torch.clamp(hold30_s - 2.0 / 60.0, min=0.0))
        newly_s1 = (hold10_s >= 0.5) & (~stage1_awarded)
        newly_s2 = (hold30_s >= 1.0) & (~stage2_awarded)
        stage1_awarded[:] |= newly_s1
        stage2_awarded[:] |= newly_s2
        r_hold = (
            1.5 * torch.clamp(hold10_s / 0.5, 0.0, 1.0)
            + 3.0 * torch.clamp(hold30_s / 1.5, 0.0, 1.0)
            + 2.0 * torch.where(has_grip & (step_in_ep >= 300), torch.clamp(min_lift / 0.032, 0.0, 1.2), 0.0)
            + 20.0 * newly_s1.float()
            + 40.0 * newly_s2.float()
        )
        r_ctrl = -0.005 * ((act - prev_act) ** 2).mean(dim=-1) - 0.01
        reward = 0.1 * (r_dist + r_contact + r_lift + r_hold + r_ctrl)

        prev_act = act.clone()
        step_in_ep[:] += 1
        return get_obs(), reward, med_lift, min_lift, f_l_pads, f_r_pads

    init_std_val = 0.025 if args.mode == "pinch" else (0.012 if args.mode == "tripod" else 0.008)
    model = build_actor_critic(67, 24, init_std=init_std_val, device=device)

    if args.eval_only:
        assert args.checkpoint, "--eval-only requires --checkpoint"
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.eval()
        reset_all()
        obs = get_obs()
        max_med = torch.zeros((N,), device=device)
        max_min = torch.zeros((N,), device=device)
        tot_ret = torch.zeros((N,), device=device)
        with torch.no_grad():
            for _ in range(args.horizon):
                mean, _, _ = model(obs)
                obs, rew, med_l, min_l, flp, frp = step_env(mean)
                tot_ret += rew
                max_med = torch.maximum(max_med, med_l)
                max_min = torch.maximum(max_min, min_l)
        print(
            f"[EVAL {args.mode}] ckpt={args.checkpoint} ret={tot_ret.mean():.2f} "
            f"s1={int(stage1_awarded.sum().item())}/{N} s2={int(stage2_awarded.sum().item())}/{N} "
            f"med={max_med.mean()*1000:.2f}mm min={max_min.mean()*1000:.2f}mm end_min={min_l.mean()*1000:.2f}mm "
            f"hold30={hold30_s.mean():.2f}s end_envs={np.round((min_l*1000).cpu().numpy(),1).tolist()} "
            f"Lf={np.round(flp.mean(0).cpu().numpy(),2).tolist()} "
            f"Rf={np.round(frp.mean(0).cpu().numpy(),2).tolist()}",
            flush=True,
        )
        sys.stdout.flush()
        os._exit(0)

    model.log_std.requires_grad_(False)
    actor_lr = 5e-7 if (args.checkpoint and args.mode == "power") else 1.5e-6
    actor_opt = torch.optim.Adam(model.actor.parameters(), lr=actor_lr)
    critic_opt = torch.optim.Adam(model.critic.parameters(), lr=1e-3)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        with torch.no_grad():
            model.log_std.fill_(float(np.log(init_std_val)))
        bc_actor = copy.deepcopy(model.actor).eval()
        for param in bc_actor.parameters():
            param.requires_grad_(False)
        print(f"[RESUME] loaded {args.checkpoint} (upd={ckpt.get('update')})", flush=True)
    else:
        # Phase A: Collect 1 rollout of clean scripted expert across N envs for BC warm-start
        reset_all()
        bc_obs, bc_act, bc_rew = [], [], []
        obs = get_obs()
        f_l_p = torch.zeros((N, 5), device=device)
        f_r_p = torch.zeros((N, 5), device=device)
        t_start = time.perf_counter()
        active_noise_mask = torch.zeros((1, 24), device=device)
        active_noise_mask[:, 1:6] = 1.0
        active_noise_mask[:, 7:12] = 1.0
        active_noise_mask[:, 12:18] = 1.0
        for t in range(args.horizon):
            exp_a = scripted_batch_action(t, f_l_p, f_r_p, args.mode, device)
            eps_n = 0.015 if args.mode == "pinch" else 0.0
            noisy_a = (exp_a + eps_n * active_noise_mask * torch.randn_like(exp_a)).clamp(-1.0, 1.0)
            bc_obs.append(obs)
            bc_act.append(exp_a)
            obs, rew_e, med_l, min_l, f_l_p, f_r_p = step_env(noisy_a)
            bc_rew.append(rew_e)
        print(f"[Expert R1] mode={args.mode} s1={int(stage1_awarded.sum().item())}/{N} s2={int(stage2_awarded.sum().item())}/{N} end_min={min_l.mean()*1000:.2f}mm", flush=True)
        bc_obs_t = torch.cat(bc_obs, dim=0)
        bc_act_t = torch.cat(bc_act, dim=0)
        bc_lr1 = 2e-3
        bc_opt = torch.optim.Adam(model.actor.parameters(), lr=bc_lr1)
        bc_sched = torch.optim.lr_scheduler.CosineAnnealingLR(bc_opt, T_max=max(1, args.bc_epochs), eta_min=1e-4)
        for ep in range(args.bc_epochs):
            perm = torch.randperm(bc_obs_t.shape[0], device=device)
            for i in range(0, bc_obs_t.shape[0], 256):
                idx = perm[i : i + 256]
                pred = (1.05 * torch.tanh(model.actor(bc_obs_t[idx]))).clamp(-1.0, 1.0)
                loss = nn.functional.mse_loss(pred, bc_act_t[idx])
                bc_opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0); bc_opt.step()
            bc_sched.step()

        # Round 2 (DAgger): rollout learned BC policy, label visited states with expert
        reset_all()
        obs = get_obs()
        f_l_p = torch.zeros((N, 5), device=device)
        f_r_p = torch.zeros((N, 5), device=device)
        dag_obs, dag_act, dag_rew = [], [], []
        for t_step in range(args.horizon):
            exp_a = scripted_batch_action(t_step, f_l_p, f_r_p, args.mode, device)
            with torch.no_grad():
                pol_a, _, _ = model(obs)
                eps_d = 0.015 if args.mode == "pinch" else 0.004
                pol_a = (pol_a + eps_d * active_noise_mask * torch.randn_like(pol_a)).clamp(-1.0, 1.0)
            dag_obs.append(obs)
            dag_act.append(exp_a)
            obs, rew_d, _, min_d, f_l_p, f_r_p = step_env(pol_a)
            dag_rew.append(rew_d)
        print(f"[DAgger R2] closed-loop stage1_passed={int(stage1_awarded.sum().item())}/{N} stage2={int(stage2_awarded.sum().item())}/{N} end_min={min_d.mean()*1000:.2f}mm", flush=True)
        bc_obs_t = torch.cat([bc_obs_t] + dag_obs, dim=0)
        bc_act_t = torch.cat([bc_act_t] + dag_act, dim=0)
        bc_rew = dag_rew
        bc_lr2 = 1.5e-3 if args.mode == "pinch" else 5e-4
        bc_opt = torch.optim.Adam(model.actor.parameters(), lr=bc_lr2)
        bc_sched = torch.optim.lr_scheduler.CosineAnnealingLR(bc_opt, T_max=max(1, args.bc_epochs), eta_min=3e-5)
        best_bc_mse = 1e9
        best_bc_sd = copy.deepcopy(model.actor.state_dict())
        for ep in range(args.bc_epochs):
            perm = torch.randperm(bc_obs_t.shape[0], device=device)
            ep_mse, ep_cnt = 0.0, 0
            for i in range(0, bc_obs_t.shape[0], 256):
                idx = perm[i : i + 256]
                pred = (1.05 * torch.tanh(model.actor(bc_obs_t[idx]))).clamp(-1.0, 1.0)
                loss = nn.functional.mse_loss(pred, bc_act_t[idx])
                bc_opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0); bc_opt.step()
                ep_mse += float(loss.item()); ep_cnt += 1
            bc_sched.step()
            ep_mse /= max(1, ep_cnt)
            if ep_mse < best_bc_mse:
                best_bc_mse = ep_mse
                best_bc_sd = copy.deepcopy(model.actor.state_dict())
        model.actor.load_state_dict(best_bc_sd)
        bc_obs_t = torch.cat(dag_obs, dim=0)
        bc_rew_t = torch.stack(bc_rew, dim=0)  # [H, N]
        bc_ret = torch.zeros_like(bc_rew_t)
        g_ret = torch.zeros((N,), device=device)
        for t_i in reversed(range(args.horizon)):
            g_ret = bc_rew_t[t_i] + 0.995 * g_ret
            bc_ret[t_i] = g_ret
        bc_ret_flat = bc_ret.reshape(-1)
        for _ in range(100):
            perm = torch.randperm(bc_obs_t.shape[0], device=device)
            for i in range(0, bc_obs_t.shape[0], 512):
                idx = perm[i : i + 512]
                v_pred = model.critic(bc_obs_t[idx]).squeeze(-1)
                v_loss = nn.functional.mse_loss(v_pred, bc_ret_flat[idx])
                critic_opt.zero_grad(); v_loss.backward(); critic_opt.step()
        bc_actor = copy.deepcopy(model.actor).eval()
        for param in bc_actor.parameters():
            param.requires_grad_(False)
        print(
            f"[BC WARM-START] transitions={bc_obs_t.shape[0]} final_mse={loss.item():.5f} "
            f"critic_mse={v_loss.item():.2f} stage1={int(stage1_awarded.sum().item())}/{N} "
            f"stage2={int(stage2_awarded.sum().item())}/{N} wall={time.perf_counter()-t_start:.1f}s",
            flush=True,
        )

    # Phase B: Multi-Env GPU PPO Training
    T = args.horizon
    gamma, lam, clip_eps = 0.995, 0.95, 0.2
    best_ret = -1e9
    metrics_path = out_dir / "metrics.jsonl"

    for upd in range(1, args.updates + 1):
        t_u = time.perf_counter()
        pre_step_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        reset_all()
        obs = get_obs()
        obs_buf = torch.zeros((T, N, 67), device=device)
        act_buf = torch.zeros((T, N, 24), device=device)
        mean_buf = torch.zeros((T, N, 24), device=device)
        logp_buf = torch.zeros((T, N), device=device)
        rew_buf = torch.zeros((T, N), device=device)
        val_buf = torch.zeros((T, N), device=device)
        max_med = torch.zeros((N,), device=device)
        max_min = torch.zeros((N,), device=device)

        with torch.no_grad():
            for t in range(T):
                mean, std, val = model(obs)
                dist = torch.distributions.Normal(mean, std)
                raw_a = dist.sample()
                a = raw_a.clamp(-1.0, 1.0)
                logp = dist.log_prob(raw_a).sum(dim=-1)
                obs_buf[t] = obs
                act_buf[t] = raw_a
                mean_buf[t] = mean
                logp_buf[t] = logp
                val_buf[t] = val
                obs, rew, med_l, min_l, _, _ = step_env(a)
                rew_buf[t] = rew
                max_med = torch.maximum(max_med, med_l)
                max_min = torch.maximum(max_min, min_l)
            _, _, last_val = model(obs)

        adv_buf = torch.zeros_like(rew_buf)
        gae = torch.zeros((N,), device=device)
        for t in reversed(range(T)):
            next_v = last_val if t == T - 1 else val_buf[t + 1]
            delta = rew_buf[t] + gamma * next_v - val_buf[t]
            gae = delta + gamma * lam * gae
            adv_buf[t] = gae
        ret_buf = adv_buf + val_buf

        b_obs = obs_buf.reshape(T * N, 67)
        b_act = act_buf.reshape(T * N, 24)
        b_mean = mean_buf.reshape(T * N, 24)
        b_logp = logp_buf.reshape(T * N)
        b_adv = adv_buf.reshape(T * N)
        b_adv = (b_adv - b_adv.mean()) / b_adv.std().clamp(min=0.1)
        b_ret = ret_buf.reshape(T * N)

        kl_val = 0.0
        stop_actor = False
        for _epoch in range(4):
            perm = torch.randperm(T * N, device=device)
            mb_size = (T * N) // 8
            for start in range(0, T * N, mb_size):
                idx = perm[start : start + mb_size]
                mean, std, val = model(b_obs[idx])
                v_loss = 0.5 * ((val - b_ret[idx]) ** 2).mean()
                critic_opt.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(model.critic.parameters(), 0.5)
                critic_opt.step()
                if not stop_actor:
                    old_actor_sd = {k: v.clone() for k, v in model.actor.state_dict().items()}
                    dist = torch.distributions.Normal(mean, std)
                    new_logp = dist.log_prob(b_act[idx]).sum(dim=-1)
                    ratio = torch.exp((new_logp - b_logp[idx]).clamp(-5.0, 5.0))
                    surr1 = ratio * b_adv[idx]
                    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv[idx]
                    p_loss = -torch.min(surr1, surr2).mean()
                    with torch.no_grad():
                        bc_mean = (1.05 * torch.tanh(bc_actor(b_obs[idx]))).clamp(-1.0, 1.0)
                    policy_kl_bc = ((mean - bc_mean) ** 2).mean()
                    bc_anchor_w = 0.05 if (args.checkpoint or args.mode == "pinch") else 0.15
                    total_p = p_loss + bc_anchor_w * policy_kl_bc
                    actor_opt.zero_grad()
                    total_p.backward()
                    nn.utils.clip_grad_norm_(model.actor.parameters(), 0.2)
                    actor_opt.step()
                    with torch.no_grad():
                        new_mean, new_std, _ = model(b_obs[idx])
                        step_kl = float((0.5 * (((new_mean - b_mean[idx]) / new_std) ** 2)).sum(dim=-1).mean().item())
                    kl_limit = 0.015 if args.mode == "pinch" else (0.012 if args.mode == "tripod" else 0.008)
                    if step_kl > kl_limit:
                        model.actor.load_state_dict(old_actor_sd)
                        stop_actor = True
                    else:
                        kl_val = step_kl

        mean_ret = float(rew_buf.sum(dim=0).mean().item())
        s1_pass = int(stage1_awarded.sum().item())
        s2_pass = int(stage2_awarded.sum().item())
        med_mm = float(max_med.mean().item() * 1000.0)
        min_mm = float(max_min.mean().item() * 1000.0)
        end_min_mm = float(min_l.mean().item() * 1000.0)
        worst_end_mm = float(min_l.min().item() * 1000.0)
        h30_s = float(hold30_s.mean().item())
        dt_u = time.perf_counter() - t_u
        row = {
            "mode": args.mode,
            "update": upd,
            "mean_return": round(mean_ret, 2),
            "stage1_passed": s1_pass,
            "stage2_passed": s2_pass,
            "num_envs": N,
            "mean_max_med_mm": round(med_mm, 2),
            "mean_max_min_mm": round(min_mm, 2),
            "end_min_mm": round(end_min_mm, 2),
            "hold30_s": round(h30_s, 2),
            "kl": round(kl_val, 5),
            "wall_s": round(dt_u, 2),
        }
        with metrics_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"[PPO upd {upd:2d}/{args.updates}] mode={args.mode} ret={mean_ret:7.2f} s1={s1_pass:2d}/{N} s2={s2_pass:2d}/{N} "
            f"med={med_mm:5.2f}mm min={min_mm:5.2f}mm end_min={end_min_mm:5.2f}mm hold30={h30_s:.2f}s kl={kl_val:.4f} wall={dt_u:.1f}s",
            flush=True,
        )
        ckpt_payload = {"model": model.state_dict(), "mode": args.mode, "update": upd, "row": row}
        torch.save(ckpt_payload, out_dir / "latest.pt")
        ckpt_score = (s2_pass * 100.0) + min(max(end_min_mm, -50.0), 38.0) * 3.0 + mean_ret
        if worst_end_mm > -200.0 and ckpt_score > best_ret:
            best_ret = ckpt_score
            torch.save({"model": pre_step_sd, "mode": args.mode, "update": upd, "row": row}, out_dir / "best.pt")

    print(f"TRAINING DONE. Best return={best_ret:.2f}, saved to {out_dir}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
