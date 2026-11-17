#!/usr/bin/env python3
"""Pure-GPU fixed-front-stereo Student BC + DAgger distillation from Teacher PPO."""
import argparse
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


def build_stereo_student(proprio_dim: int = 39, act_dim: int = 24, embed_dim: int = 128, device: str = "cuda:0"):
    import torch
    import torch.nn as nn
    from go2_pvcnn.control.dh116.tabletop_stereo_encoder import StereoEncoder

    class StereoStudentNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = StereoEncoder(embed_dim=embed_dim, num_heads=4, image_processing_batch_size=None)
            self.proprio_proj = nn.Sequential(
                nn.Linear(proprio_dim, embed_dim),
                nn.ELU(),
                nn.Linear(embed_dim, embed_dim),
            )
            self.cross_attn = nn.MultiheadAttention(embed_dim, 4, dropout=0.0, batch_first=True)
            self.gate = nn.Linear(2 * embed_dim, embed_dim)
            self.norm = nn.LayerNorm(embed_dim)
            self.head = nn.Sequential(
                nn.Linear(embed_dim, 256),
                nn.ELU(),
                nn.Linear(256, 256),
                nn.ELU(),
                nn.Linear(256, act_dim),
            )

        def forward(self, left_rgb, right_rgb, proprio):
            b = left_rgb.shape[0]
            valid = torch.ones((b, 2), dtype=torch.bool, device=left_rgb.device)
            tokens, tok_valid = self.encoder(left_rgb, right_rgb, valid)
            q = self.proprio_proj(proprio)
            vis = self.cross_attn(q.unsqueeze(1), tokens, tokens, key_padding_mask=~tok_valid, need_weights=False)[0][:, 0]
            g = torch.sigmoid(self.gate(torch.cat((q, vis), dim=-1)))
            feat = self.norm(q + g * vis)
            return (1.05 * torch.tanh(self.head(feat))).clamp(-1.0, 1.0)

    return StereoStudentNet().to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-ckpt", type=str, required=True)
    parser.add_argument("--mode", type=str, default=None, choices=["pinch", "tripod", "power"])
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=480)
    parser.add_argument("--dagger-rounds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch as _torch_pre
    _t_meta = _torch_pre.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
    mode = args.mode or _t_meta.get("mode", "pinch")

    from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as builder
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_scene_preparation as scene_api
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS, MIMIC, active_limits, expand
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import box_sdf
    from go2_pvcnn.control.dh116.tabletop_dual_hand_rod_runtime import PADS
    from go2_pvcnn.control.dh116.tabletop_preview_warp_readback import isaac45_rgb_readback_scope
    from go2_pvcnn.control.dh116.tabletop_tiled_stereo_native import author_standard_camera_ops, prepare_renderer_view
    from isaaclab.app import AppLauncher
    from scripts.train_dh116_gpu_teacher_ppo import build_actor_critic

    sys.path.insert(0, str(Path(builder.BACKEND).parents[1]))
    from ropesim import RopeBatch, sdf

    app = AppLauncher(headless=True, enable_cameras=True).app

    import isaaclab.sim as sim_utils
    import torch
    import torch.nn as nn
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.sensors import TiledCamera, TiledCameraCfg
    from isaaclab.utils.math import quat_from_matrix
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt

    from go2_pvcnn.control.dh116.tabletop_material import hand_material_contract
    from go2_pvcnn.control.dh116.tabletop_palm_mount import (
        ANGULAR_JOINT_NAMES,
        CONTRACT as WRIST,
        JOINT_NAMES,
        ROTATION_CONTRACT,
        local_translation,
    )
    from go2_pvcnn.control.dh116.tabletop_vec_isaac import spawn_scene
    from go2_pvcnn.control.dh116.tabletop_velocity_limits import ACTIVE_VELOCITY_LIMIT_RAD_S, passive_velocity_limit

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = "cuda:0"
    N = int(args.num_envs)
    out_dir = ROOT / "runs/gpu_stereo_student" / f"s{N}-{mode}-{uuid.uuid4().hex[:8]}"
    if not args.eval_only:
        out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INIT] mode={mode}, num_envs={N}, horizon={args.horizon}, rounds={args.dagger_rounds}, out={out_dir}", flush=True)

    sim_cfg = sim_utils.SimulationCfg(
        dt=DT, device=device, render_interval=1, gravity=(0.0, 0.0, -9.81),
        physx=sim_utils.PhysxCfg(solver_type=0),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    prepared = scene_api.prepare_scene(ROOT)
    sp_args, layout, sources = scene_api.spawn_parameters(ROOT, prepared)
    UsdGeom.Xform.Define(sim.stage, "/World/envs")
    light = UsdLux.DomeLight.Define(sim.stage, "/World/DomeLight")
    light.CreateIntensityAttr(900.0)
    light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

    cam_cfg = json.loads((ROOT / "configs/dh116_training/stereo_camera_front_640x480_v3.json").read_text())
    W, H = int(cam_cfg["width"]), int(cam_cfg["height"])
    K_l = np.asarray(cam_cfg["K_left"], dtype=np.float32)
    focal = 24.0
    env_origins_np = np.array([[0.0, float(i) * 6.0, 0.0] for i in range(N)], dtype=np.float32)
    env_origins = torch.tensor(env_origins_np, dtype=torch.float32, device=device)

    for i in range(N):
        prefix = f"/World/envs/env_{i}"
        spawn_scene(sim, sp_args, layout, sources, prefix, env_origins_np[i], external_wire=True)
        for p in Usd.PrimRange(sim.stage.GetPrimAtPath(f"{prefix}/Table")):
            if p.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Set(False)
        for cam_name in ("StereoLeft", "StereoRight"):
            cam = UsdGeom.Camera.Define(sim.stage, f"{prefix}/{cam_name}")
            author_standard_camera_ops(cam, UsdGeom, Gf)
            cam.CreateProjectionAttr("perspective")
            cam.CreateFocalLengthAttr(focal)
            cam.CreateHorizontalApertureAttr(W * focal / float(K_l[0, 0]))
            cam.CreateVerticalApertureAttr(H * focal / float(K_l[1, 1]))
            cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10.0))

    import carb.settings
    carb.settings.get_settings().set_bool("/app/player/playSimulations", False)

    class ExplicitTiledCamera(TiledCamera):
        def _create_buffers(self):
            prepare_renderer_view(self._view)
            return super()._create_buffers()

    left_cam = ExplicitTiledCamera(TiledCameraCfg(
        prim_path="/World/envs/env_.*/StereoLeft", spawn=None, width=W, height=H,
        data_types=["rgb"], update_period=0.0, update_latest_camera_pose=True, debug_vis=False,
    ))
    right_cam = ExplicitTiledCamera(TiledCameraCfg(
        prim_path="/World/envs/env_.*/StereoRight", spawn=None, width=W, height=H,
        data_types=["rgb"], update_period=0.0, update_latest_camera_pose=True, debug_vis=False,
    ))

    def _make_art(side: str):
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
        return Articulation(ArticulationCfg(
            prim_path=f"/World/envs/env_.*/{side}Hand", spawn=None,
            init_state=ArticulationCfg.InitialStateCfg(joint_pos=dict(zip(JOINTS, map(float, expand(layout["active_joints"][side]))))),
            actuators=actuators,
        ))

    left_hands = _make_art("left")
    right_hands = _make_art("right")
    sim.reset()
    left_hands.update(DT)
    right_hands.update(DT)

    for cam, T_key in ((left_cam, "T_world_left"), (right_cam, "T_world_right")):
        T = torch.tensor(cam_cfg[T_key], dtype=torch.float32, device=device)
        pos = T[:3, 3][None] + env_origins
        quat = quat_from_matrix(T[:3, :3])[None].repeat(N, 1)
        cam.set_world_poses(positions=pos, orientations=quat, convention="ros")

    cfg_rod = json.loads((ROOT / "configs/dh116_tasks/flexible_pinch_undamped_v1.json").read_text())
    r_cfg = cfg_rod["rod"]
    if mode == "pinch":
        pair_packet = builder.load_pair_proposal()
        seed = pair_packet["reset_seed"]
        base_nodes = np.asarray(pair_packet["rod_seed"]["proposed_nodes_m"], dtype=np.float32)
        seed_joint_11 = [seed["joint_positions"][0], seed["joint_positions"][1]]
        seed_root_list = seed["reset_root_pose_w"]
    else:
        style_cfg = json.loads((ROOT / "configs/dh116_training/gpu_teacher_style_seeds_v1.json").read_text())
        s_entry = style_cfg["styles"][mode]
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
    rope = RopeBatch.from_positions(
        base_nodes[None] + env_origins_np[:, None],
        density=r_cfg["linear_density_kg_m"], k_stretch=r_cfg["axial_stiffness_n"], k_bend=r_cfg["bend_stiffness_n_m2"],
        max_h=DT / r_cfg["solver_substeps"], num_itr=r_cfg["solver_iterations"],
        pinned=np.zeros((N, len(base_nodes)), dtype=bool), fix_root_orientation=False, straight_rest=True, deterministic=True,
    )
    rope.set_gravity(np.asarray(r_cfg["gravity_world_m_s2"], dtype=np.float32))
    rope.set_contact(radius=r_cfg["radius_m"], mu_s=r_cfg["mu_s"], mu_k=r_cfg["mu_k"], v_slip=r_cfg["v_slip_m_s"], kappa=r_cfg["kappa"])
    rope.set_rope_contact(False)
    rope.damping = float(r_cfg["damping"])
    rope.drag = float(r_cfg["drag"])

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

    seed_q = torch.zeros((2, 17), device=device, dtype=torch.float32)
    for hi, h in enumerate((left_hands, right_hands)):
        for j, name in enumerate(JOINTS):
            seed_q[hi, h.joint_names.index(name)] = float(seed_joint_11[hi][j])
    seed_root_base = torch.tensor(seed_root_list, device=device, dtype=torch.float32)

    hand_mat = hand_material_contract(cfg_rod["hand"]["material_profile"])["hand_contact_material"]
    pad_mu_s = max(r_cfg["mu_s"], hand_mat["static_friction"])
    pad_mu_k = max(r_cfg["mu_k"], hand_mat["dynamic_friction"])

    all_env_ids_np = np.arange(N, dtype=np.int32)
    for hi, (h, r_pose) in enumerate(((left_hands, seed_root_base[0]), (right_hands, seed_root_base[1]))):
        root_w = r_pose.unsqueeze(0).repeat(N, 1); root_w[:, :3] += env_origins
        q_n = seed_q[hi : hi + 1].repeat(N, 1)
        h.write_root_pose_to_sim(root_w); h.write_root_velocity_to_sim(torch.zeros((N, 6), device=device))
        h.write_joint_state_to_sim(q_n, torch.zeros_like(q_n)); h.set_joint_position_target(q_n)
        h.write_data_to_sim(); h.update(DT)

    for i, name in enumerate(PADS):
        rope.add_mesh(*grids["left." + name].as_args(),
            pos=left_hands.data.body_link_pos_w[:, l_pad_ids[i]].cpu().numpy(),
            quat=left_hands.data.body_link_quat_w[:, l_pad_ids[i]][:, [1, 2, 3, 0]].cpu().numpy(),
            mu_s=pad_mu_s, mu_k=pad_mu_k)
    table_size = np.asarray(prepared["table"]["size_m"], dtype=np.float32)
    table_center = np.asarray(prepared["table"]["center_m"], dtype=np.float32)
    half = table_size / 2
    tbl_cfg = cfg_rod["table"]
    tbl_grid = sdf.from_function(lambda p: box_sdf(p, table_size), -half - tbl_cfg["sdf_margin_m"], half + tbl_cfg["sdf_margin_m"], tbl_cfg["sdf_spacing_m"])
    tbl_pos = table_center[None, :] + env_origins_np
    tbl_quat = np.zeros((N, 4), dtype=np.float32); tbl_quat[:, 3] = 1.0
    rope.add_mesh(*tbl_grid.as_args(), pos=tbl_pos, quat=tbl_quat, mu_s=tbl_cfg["mu_s"], mu_k=tbl_cfg["mu_k"])
    for i, name in enumerate(PADS):
        rope.add_mesh(*grids["right." + name].as_args(),
            pos=right_hands.data.body_link_pos_w[:, r_pad_ids[i]].cpu().numpy(),
            quat=right_hands.data.body_link_quat_w[:, r_pad_ids[i]][:, [1, 2, 3, 0]].cpu().numpy(),
            mu_s=pad_mu_s, mu_k=pad_mu_k)
    rope.enable_contact_forces(True)

    rope_pos_view = torch.as_tensor(rope.positions())
    wrench_view = torch.as_tensor(rope.mesh_wrenches_view())
    mesh_pos = torch.zeros((N, 11, 3), dtype=torch.float32, device=device)
    mesh_quat = torch.zeros((N, 11, 4), dtype=torch.float32, device=device)
    mesh_pos[:, 5] = torch.tensor(tbl_pos, dtype=torch.float32, device=device)
    mesh_quat[:, 5, 3] = 1.0
    tele_buf = torch.zeros((N,), dtype=torch.bool, device=device)

    # Combined USD Mesh for visual rendering of the N flexible Cosserat wires
    SIDES, NODES = 10, len(base_nodes)
    wire_mesh = UsdGeom.Mesh.Define(sim.stage, "/World/CosseratWires")
    mat_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.35, 0.02), roughness=0.5)
    mat_cfg.func("/World/CosseratWireMat", mat_cfg)
    UsdShade.MaterialBindingAPI(wire_mesh).Bind(UsdShade.Material(sim.stage.GetPrimAtPath("/World/CosseratWireMat")))
    counts, indices = [], []
    for e in range(N):
        bv = e * NODES * SIDES
        for s in range(NODES - 1):
            for j in range(SIDES):
                nj = (j + 1) % SIDES
                counts.append(4)
                indices.extend([bv + s*SIDES + j, bv + s*SIDES + nj, bv + (s+1)*SIDES + nj, bv + (s+1)*SIDES + j])
    wire_mesh.CreateFaceVertexCountsAttr().Set(counts)
    wire_mesh.CreateFaceVertexIndicesAttr().Set(indices)
    pts_attr = wire_mesh.CreatePointsAttr()
    ext_attr = wire_mesh.CreateExtentAttr()
    angles = torch.arange(SIDES, device=device, dtype=torch.float32) * (2.0 * np.pi / SIDES)
    cos_a = torch.cos(angles)[None, None, :, None]
    sin_a = torch.sin(angles)[None, None, :, None]
    ref_up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)
    ref_x = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32)

    def sync_wire_usd():
        p = rope_pos_view
        tang = torch.empty_like(p)
        tang[:, :-1] = p[:, 1:] - p[:, :-1]; tang[:, -1] = tang[:, -2]
        tang = tang / torch.linalg.norm(tang, dim=-1, keepdim=True).clamp_min(1e-6)
        n1 = torch.cross(tang, ref_up.expand_as(tang), dim=-1)
        n1_norm = torch.linalg.norm(n1, dim=-1, keepdim=True)
        alt = torch.cross(tang, ref_x.expand_as(tang), dim=-1)
        n1 = torch.where(n1_norm > 1e-3, n1 / n1_norm.clamp_min(1e-6), alt / torch.linalg.norm(alt, dim=-1, keepdim=True).clamp_min(1e-6))
        n2 = torch.cross(tang, n1, dim=-1)
        verts = p[:, :, None, :] + float(r_cfg["radius_m"]) * (cos_a * n1[:, :, None, :] + sin_a * n2[:, :, None, :])
        v_np = verts.reshape(-1, 3).cpu().numpy()
        pts_attr.Set(Vt.Vec3fArray.FromNumpy(v_np))
        ext_attr.Set(Vt.Vec3fArray([Gf.Vec3f(*[float(x) for x in v_np.min(0)]), Gf.Vec3f(*[float(x) for x in v_np.max(0)])]))

    lo_np, hi_np = active_limits()
    q_lo = torch.tensor(lo_np, device=device, dtype=torch.float32)
    q_hi = torch.tensor(hi_np, device=device, dtype=torch.float32)
    mimic_map = {f: (ACTIVE.index(m), ratio, off) for f, m, ratio, off in MIMIC}
    active_idx = {n: i for i, n in enumerate(ACTIVE)}

    def _expand_11(q_act):
        return torch.stack([
            q_act[:, active_idx[n]] if n in active_idx else mimic_map[n][1] * q_act[:, mimic_map[n][0]] + mimic_map[n][2]
            for n in JOINTS
        ], dim=-1)

    l_q = torch.zeros((N, 6), device=device, dtype=torch.float32)
    r_q = torch.zeros((N, 6), device=device, dtype=torch.float32)
    l_wp = torch.zeros((N, 3), device=device, dtype=torch.float32)
    r_wp = torch.zeros((N, 3), device=device, dtype=torch.float32)
    l_wa = torch.zeros((N, 3), device=device, dtype=torch.float32)
    r_wa = torch.zeros((N, 3), device=device, dtype=torch.float32)
    step_in_ep = torch.zeros((N,), device=device, dtype=torch.long)

    init_med_z = float(np.median(base_nodes[:, 2]))
    init_min_z = float(np.min(base_nodes[:, 2]))
    wire_ref_center = torch.tensor([0.43, 0.0, 0.308], device=device, dtype=torch.float32)
    wire_nodes_idx = torch.tensor([0, 10, 20, 30, 40], device=device, dtype=torch.long)

    def reset_all():
        nonlocal l_q, r_q, l_wp, r_wp, l_wa, r_wa, step_in_ep
        for hi, (h, r_pose) in enumerate(((left_hands, seed_root_base[0]), (right_hands, seed_root_base[1]))):
            root_w = r_pose.unsqueeze(0).repeat(N, 1); root_w[:, :3] += env_origins
            q_n = seed_q[hi : hi + 1].repeat(N, 1)
            h.permanent_wrench_composer.reset()
            h.write_root_pose_to_sim(root_w); h.write_root_velocity_to_sim(torch.zeros((N, 6), device=device))
            h.write_joint_state_to_sim(q_n, torch.zeros_like(q_n)); h.set_joint_position_target(q_n)
            h.write_data_to_sim(); h.update(DT)
        rope.reset_envs(all_env_ids_np)
        mesh_pos[:, 0:5] = left_hands.data.body_link_pos_w[:, l_pad_ids]
        mesh_quat[:, 0:5] = left_hands.data.body_link_quat_w[:, l_pad_ids][:, :, [1, 2, 3, 0]]
        mesh_pos[:, 6:11] = right_hands.data.body_link_pos_w[:, r_pad_ids]
        mesh_quat[:, 6:11] = right_hands.data.body_link_quat_w[:, r_pad_ids][:, :, [1, 2, 3, 0]]
        tele_buf.fill_(True)
        rope.set_mesh_poses_all(mesh_pos, mesh_quat, tele_buf)
        l_q = left_hands.data.joint_pos[:, l_act_ids].clone()
        r_q = right_hands.data.joint_pos[:, r_act_ids].clone()
        l_wp = left_hands.data.body_link_pos_w[:, l_base_id].clone()
        r_wp = right_hands.data.body_link_pos_w[:, r_base_id].clone()
        l_wa.zero_(); r_wa.zero_(); step_in_ep.zero_()
        sync_wire_usd()
        for _ in range(2):
            sim.render()
            with isaac45_rgb_readback_scope():
                left_cam.update(0.0, force_recompute=True); right_cam.update(0.0, force_recompute=True)

    def get_teacher_and_student_obs():
        l_qd = left_hands.data.joint_vel[:, l_act_ids]
        r_qd = right_hands.data.joint_vel[:, r_act_ids]
        l_bp = left_hands.data.body_link_pos_w[:, l_base_id] - env_origins - wire_ref_center
        r_bp = right_hands.data.body_link_pos_w[:, r_base_id] - env_origins - wire_ref_center
        l_bq = left_hands.data.body_link_quat_w[:, l_base_id]
        r_bq = right_hands.data.body_link_quat_w[:, r_base_id]
        t_norm = (step_in_ep.float() / float(args.horizon)).unsqueeze(-1)
        proprio = torch.cat((l_q, r_q, l_qd * 0.5, r_qd * 0.5, l_bp * 5.0, r_bp * 5.0, l_bq, r_bq, t_norm), dim=-1)  # 39D

        pad_f = torch.cat((
            torch.linalg.norm(wrench_view[:, 0:5, :3], dim=-1),
            torch.linalg.norm(wrench_view[:, 6:11, :3], dim=-1),
        ), dim=-1).clamp(0.0, 20.0) / 5.0
        w_nodes = (rope_pos_view[:, wire_nodes_idx, :] - env_origins.unsqueeze(1) - wire_ref_center).reshape(N, 15)
        z = rope_pos_view[:, :, 2]
        med_l = torch.median(z, dim=-1).values - init_med_z
        min_l = torch.min(z, dim=-1).values - init_min_z
        z_span = torch.max(z, dim=-1).values - torch.min(z, dim=-1).values
        w_stats = torch.stack((med_l * 20.0, min_l * 20.0, z_span * 20.0), dim=-1)
        t_obs = torch.cat((l_q, r_q, l_qd * 0.5, r_qd * 0.5, l_bp * 5.0, r_bp * 5.0, l_bq, r_bq, pad_f, w_nodes * 5.0, w_stats, t_norm), dim=-1)

        sync_wire_usd()
        sim.render()
        with isaac45_rgb_readback_scope():
            left_cam.update(0.0, force_recompute=True); right_cam.update(0.0, force_recompute=True)
        l_rgb = left_cam.data.output["rgb"].permute(0, 3, 1, 2).contiguous()   # [N, 3, H, W] uint8 CUDA
        r_rgb = right_cam.data.output["rgb"].permute(0, 3, 1, 2).contiguous()
        return t_obs, proprio, l_rgb, r_rgb

    def step_physics(act):
        nonlocal l_q, r_q, l_wp, r_wp, l_wa, r_wa
        act = act.clamp(-1.0, 1.0)
        for _ in range(SUBSTEPS_PER_CTRL):
            for h, pids, sl in ((left_hands, l_pad_ids, slice(0, 5)), (right_hands, r_pad_ids, slice(6, 11))):
                f = wrench_view[:, sl, :3]
                tau = wrench_view[:, sl, 3:] + torch.cross(h.data.body_link_pos_w[:, pids] - h.data.body_com_pos_w[:, pids], f, dim=-1)
                h.permanent_wrench_composer.reset()
                h.permanent_wrench_composer.set_forces_and_torques(forces=f, torques=tau, body_ids=pids, is_global=True)
            l_q = torch.clamp(l_q + act[:, 0:6] * (0.3 * DT), q_lo, q_hi)
            r_q = torch.clamp(r_q + act[:, 6:12] * (0.3 * DT), q_lo, q_hi)
            l_wp = l_wp + act[:, 12:15] * (0.01 * DT)
            r_wp = r_wp + act[:, 15:18] * (0.01 * DT)
            l_wa = torch.clamp(l_wa + act[:, 18:21] * (0.5 * DT), -1.5, 1.5)
            r_wa = torch.clamp(r_wa + act[:, 21:24] * (0.5 * DT), -1.5, 1.5)
            left_hands.set_joint_position_target(_expand_11(l_q), joint_ids=l_all_ids)
            right_hands.set_joint_position_target(_expand_11(r_q), joint_ids=r_all_ids)
            left_hands.set_joint_position_target(torch.cat((local_translation(l_wp, left_hands.data.root_pose_w), l_wa), -1), joint_ids=l_wrist_ids)
            right_hands.set_joint_position_target(torch.cat((local_translation(r_wp, right_hands.data.root_pose_w), r_wa), -1), joint_ids=r_wrist_ids)
            left_hands.write_data_to_sim(); right_hands.write_data_to_sim()
            sim.step(render=False)
            left_hands.update(DT); right_hands.update(DT)
            mesh_pos[:, 0:5] = left_hands.data.body_link_pos_w[:, l_pad_ids]
            mesh_quat[:, 0:5] = left_hands.data.body_link_quat_w[:, l_pad_ids][:, :, [1, 2, 3, 0]]
            mesh_pos[:, 6:11] = right_hands.data.body_link_pos_w[:, r_pad_ids]
            mesh_quat[:, 6:11] = right_hands.data.body_link_quat_w[:, r_pad_ids][:, :, [1, 2, 3, 0]]
            tele_buf.zero_()
            rope.set_mesh_poses_all(mesh_pos, mesh_quat, tele_buf)
            rope.step_free(DT, 12)
        step_in_ep[:] += 1
        z = rope_pos_view[:, :, 2]
        med_l = torch.median(z, dim=-1).values - init_med_z
        min_l = torch.min(z, dim=-1).values - init_min_z
        f_l = torch.linalg.norm(wrench_view[:, 0:5, :3], dim=-1).sum(dim=-1)
        f_r = torch.linalg.norm(wrench_view[:, 6:11, :3], dim=-1).sum(dim=-1)
        return med_l, min_l, f_l, f_r

    # Load frozen Teacher
    teacher = build_actor_critic(67, 24, 0.025, device)
    t_ckpt = torch.load(args.teacher_ckpt, map_location=device, weights_only=False)
    teacher.load_state_dict(t_ckpt["model"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"Loaded Teacher {args.teacher_ckpt} (mode={mode}, upd={t_ckpt['update']})", flush=True)

    student = build_stereo_student(39, 24, 128, device)
    act_weights = torch.ones((1, 24), device=device, dtype=torch.float32)
    act_weights[:, :12] = 2.0  # emphasize precise bilateral finger aperture
    act_weights[:, 12:18] = 1.5  # wrist XYZ trajectories (seating + lift)

    active_noise_mask = torch.ones((1, 24), device=device, dtype=torch.float32)
    if mode != "pinch":
        active_noise_mask[:, 0] = 0.0
        active_noise_mask[:, 6] = 0.0
        active_noise_mask[:, 18:24] = 0.0

    # Host-RAM replay storage for full-res uint8 stereo pairs
    buf_l_rgb, buf_r_rgb, buf_prop, buf_act = [], [], [], []
    best_score = (-1, -1, -1.0)

    def rollout_and_collect(use_student: bool, round_idx: int, noise_std: float = 0.015, tag: str = ""):
        reset_all()
        hold10 = torch.zeros(N, device=device)
        hold30 = torch.zeros(N, device=device)
        max_med = torch.zeros(N, device=device)
        max_min = torch.zeros(N, device=device)
        l_list, r_list, p_list, a_list = [], [], [], []
        t0 = time.perf_counter()
        for t in range(args.horizon):
            t_obs, proprio, l_rgb_u8, r_rgb_u8 = get_teacher_and_student_obs()
            with torch.no_grad():
                t_act, _, _ = teacher(t_obs)
                if use_student:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        s_act = student(l_rgb_u8.float() / 255.0, r_rgb_u8.float() / 255.0, proprio).float()
                    exec_act = s_act
                else:
                    exec_act = (t_act + noise_std * active_noise_mask * torch.randn_like(t_act)).clamp(-1.0, 1.0)
            l_list.append(l_rgb_u8.cpu())
            r_list.append(r_rgb_u8.cpu())
            p_list.append(proprio.cpu())
            a_list.append(t_act.cpu())
            med_l, min_l, f_l, f_r = step_physics(exec_act)
            max_med = torch.maximum(max_med, med_l)
            max_min = torch.maximum(max_min, min_l)
            bilat = (f_l > 0.3) & (f_r > 0.3)
            hold10 += (bilat & (min_l >= 0.01)).float() / 60.0
            hold30 += (bilat & (min_l >= 0.03)).float() / 60.0
            if use_student and t == args.horizon - 1:
                from PIL import Image
                fname = f"student_end_{tag or f'r{round_idx}'}.png"
                Image.fromarray(l_rgb_u8[0].permute(1, 2, 0).cpu().numpy()).save(out_dir / fname)
        dt_r = time.perf_counter() - t0
        if not tag:
            buf_l_rgb.append(torch.cat(l_list, dim=0))
            buf_r_rgb.append(torch.cat(r_list, dim=0))
            buf_prop.append(torch.cat(p_list, dim=0))
            buf_act.append(torch.cat(a_list, dim=0))
        s1 = int((hold10 >= 0.5).sum().item())
        s2 = int((hold30 >= 1.0).sum().item())
        print(
            f"[COLLECT R{round_idx} {'STUDENT' if use_student else 'TEACHER'}{(' '+tag) if tag else ''}] "
            f"s1={s1}/{N} s2={s2}/{N} max_med={max_med.mean()*1000:.2f}mm max_min={max_min.mean()*1000:.2f}mm "
            f"end_min={min_l.mean()*1000:.2f}mm h10={hold10.mean():.2f}s h30={hold30.mean():.2f}s wall={dt_r:.1f}s",
            flush=True,
        )
        return float(hold30.mean().item()), s1, s2, float(min_l.mean().item() * 1000.0)

    def train_epochs(round_idx: int, lr_start: float):
        student.train()
        opt = torch.optim.AdamW(student.parameters(), lr=lr_start, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
        all_l = torch.cat(buf_l_rgb, dim=0)
        all_r = torch.cat(buf_r_rgb, dim=0)
        all_p = torch.cat(buf_prop, dim=0)
        all_a = torch.cat(buf_act, dim=0)
        M = all_l.shape[0]
        t0 = time.perf_counter()
        last_loss = 0.0
        for ep in range(args.epochs):
            perm = torch.randperm(M)
            ep_loss = 0.0
            nb = 0
            for i in range(0, M, args.batch_size):
                idx = perm[i : i + args.batch_size]
                if len(idx) < 2:
                    continue
                l_b = all_l[idx].to(device, non_blocking=True).float() / 255.0
                r_b = all_r[idx].to(device, non_blocking=True).float() / 255.0
                p_b = all_p[idx].to(device, non_blocking=True)
                a_b = all_a[idx].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred = student(l_b, r_b, p_b)
                    loss = (act_weights * (pred - a_b) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt.step()
                ep_loss += float(loss.item()); nb += 1
            sched.step()
            last_loss = ep_loss / max(1, nb)
        print(f"[TRAIN R{round_idx}] samples={M} epochs={args.epochs} lr={lr_start:.1e}->1e-5 final_wmse={last_loss:.5f} wall={time.perf_counter()-t0:.1f}s", flush=True)
        return last_loss

    # Round 0: Teacher Stereo RGB rollouts + cosine BC
    if args.eval_only:
        assert args.checkpoint, "--eval-only requires --checkpoint"
        ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
        student.load_state_dict(ck["student"])
        student.eval()
        h30_e, s1_e, s2_e, end_e = rollout_and_collect(use_student=True, round_idx=ck.get("round", 0), tag="eval")
        print(f"EVAL DONE. mode={mode} s1={s1_e}/{N} s2={s2_e}/{N} end_min={end_e:.2f}mm hold30={h30_e:.2f}s", flush=True)
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0)

    if mode == "pinch":
        rollout_and_collect(use_student=False, round_idx=0, noise_std=0.015)
        rollout_and_collect(use_student=False, round_idx=0, noise_std=0.025)
    else:
        rollout_and_collect(use_student=False, round_idx=0, noise_std=0.0)
        rollout_and_collect(use_student=False, round_idx=0, noise_std=0.006)
    train_epochs(0, lr_start=args.lr)

    # Rounds 1..dagger_rounds: Closed-loop Student DAgger
    for r in range(1, args.dagger_rounds + 1):
        student.eval()
        h30, s1, s2, end_min = rollout_and_collect(use_student=True, round_idx=r)
        score = (s2, s1, min(end_min, 35.0), h30)
        if score > best_score:
            best_score = score
            torch.save({"student": student.state_dict(), "mode": mode, "round": r, "hold30_s": h30, "s1": s1, "s2": s2, "end_min_mm": end_min}, out_dir / "best_student.pt")
        train_epochs(r, lr_start=0.5 * args.lr)

    # Final eval before saving + reload best checkpoint for verified deterministic eval
    student.eval()
    h30, s1, s2, end_min = rollout_and_collect(use_student=True, round_idx=args.dagger_rounds + 1, tag="last")
    score = (s2, s1, min(end_min, 35.0), h30)
    if score > best_score:
        best_score = score
        torch.save({"student": student.state_dict(), "mode": mode, "round": args.dagger_rounds + 1, "hold30_s": h30, "s1": s1, "s2": s2, "end_min_mm": end_min}, out_dir / "best_student.pt")
    best_ckpt = torch.load(out_dir / "best_student.pt", map_location=device, weights_only=False)
    student.load_state_dict(best_ckpt["student"])
    student.eval()
    h30_b, s1_b, s2_b, end_b = rollout_and_collect(use_student=True, round_idx=best_ckpt["round"], tag="best")
    print(f"STUDENT DONE. mode={mode} best(R{best_ckpt['round']}) s1={s1_b}/{N} s2={s2_b}/{N} end_min={end_b:.2f}mm hold30={h30_b:.2f}s saved to {out_dir}", flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
