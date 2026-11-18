"""Independent supported-scene wiring around the frozen Teacher environment.

Only extra static collision shapes, rigid Z seed translation and explicit settled
Lift baseline are added. Original policy/control/actuator code is not edited.
"""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

from .tabletop_teacher_grasp_preview_runtime import compile_runtime,load_seeds,MODES


def compile_supported_runtime(source,expected_sha):
    tree=compile_runtime(source,expected_sha)
    factory=tree.body[0]
    factory.args.args.append(ast.arg(arg="support_plan"))
    out=[];counts=dict(bind=0,buffers=0,reference=0)
    for node in factory.body:
        if (isinstance(node,ast.Expr) and isinstance(node.value,ast.Call)
                and isinstance(node.value.func,ast.Attribute)
                and isinstance(node.value.func.value,ast.Name)
                and node.value.func.value.id=="rope"
                and node.value.func.attr=="enable_contact_forces"):
            out.extend(ast.parse("""support_binding=bind_native_end_supports(
    rope,sim.stage,"/World/envs",support_plan["scene"],env_origins_np,
    spacing_m=tbl_cfg["sdf_spacing_m"],margin_m=tbl_cfg["sdf_margin_m"],
    mu_s=tbl_cfg["mu_s"],mu_k=tbl_cfg["mu_k"])
""").body)
            counts["bind"]+=1
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="tele_buf" for t in node.targets):
            out.extend(ast.parse("""mesh_pos,mesh_quat=extend_mesh_pose_buffers(
    mesh_pos,mesh_quat,support_plan["scene"],env_origins_np)
""").body)
            counts["buffers"]+=1
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="wire_ref_center" for t in node.targets):
            node=ast.parse("""wire_ref_center=torch.tensor(
    [0.43,0.0,0.308+support_plan["scene"]["dz_m"]],
    device=device,dtype=torch.float32)
""").body[0]
            counts["reference"]+=1
        if isinstance(node,ast.Return):
            out.extend(ast.parse("""@torch.inference_mode()
def set_settled_lift_reference():
    nonlocal init_med_z,init_min_z
    z=rope_pos_view[:,:,2]
    init_med_z=torch.median(z,dim=-1).values.detach().clone()
    init_min_z=torch.min(z,dim=-1).values.detach().clone()
    step_in_ep.zero_()
    prev_act.zero_()
    hold10_s.zero_()
    hold30_s.zero_()
    stage1_awarded.zero_()
    stage2_awarded.zero_()
    return dict(initial_min_z_m=init_min_z.detach().cpu().tolist(),
        initial_median_z_m=init_med_z.detach().cpu().tolist(),
        sdk_step=sim.current_time_step_index,sdk_time_s=float(sim.current_time),
        lift_reference="actual_settled_rod_height")
""").body)
        out.append(node)
    if counts!={"bind":1,"buffers":1,"reference":1}:
        raise ValueError("supported runtime source wiring differs: "+str(counts))
    factory.body=out
    return ast.fix_missing_locations(tree)


def supported_seed_plan(root):
    import numpy as np
    from .tabletop_deformx_end_supported_scene import plan_flexible_end_supported_scene
    root=Path(root)
    config=json.loads((root/"configs/dh116_tasks/flexible_teacher_end_supported_v1.json").read_text())
    reference=json.loads((root/"configs/dh116_training/pose_reference.json").read_text())["reference_layout"]
    task=json.loads((root/config["base_task"]).read_text())
    seeds=dict(zip(MODES,load_seeds(root)))
    nodes=np.asarray(seeds["pinch"]["nodes"],dtype=float)
    layout=dict(reference)
    layout.update(wire_center=nodes.mean(axis=0),
        hand_roots={side:np.asarray(seeds["pinch"]["roots"][i],dtype=float)
            for i,side in enumerate(("left","right"))},wire_axis="Y")
    plan=plan_flexible_end_supported_scene(layout,seeds,config["clearance_m"],
        wire_length_m=float(np.linalg.norm(np.diff(nodes,axis=0),axis=1).sum()),
        wire_radius_m=float(task["rod"]["radius_m"]))
    return plan,config


def create_supported_runtime(root,args,app,source_sha):
    import importlib.util
    from .tabletop_deformx_end_supported_scene import bind_native_end_supports,extend_mesh_pose_buffers
    root=Path(root);path=root/"scripts/train_dh116_gpu_teacher_ppo.py"
    code=compile_supported_runtime(path.read_text(),source_sha)
    plan,config=supported_seed_plan(root)
    spec=importlib.util.spec_from_file_location("_dh116_supported_preview_teacher",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    namespace=dict(module.__dict__,bind_native_end_supports=bind_native_end_supports,
        extend_mesh_pose_buffers=extend_mesh_pose_buffers)
    exec(compile(code,str(path)+"[supported-preview-only]","exec"),namespace)
    values=namespace["create_runtime"](args,app,
        [plan["seeds_by_style"][style] for style in MODES],plan)
    values["supported_scene_plan"]=plan;values["supported_scene_config"]=config
    return SimpleNamespace(**values)
