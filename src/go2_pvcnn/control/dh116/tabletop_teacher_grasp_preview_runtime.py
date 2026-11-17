"""Read-only adaptation of the existing Teacher rollout for a three-grasp viewer."""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

MODES=("pinch","tripod","power")


def load_seeds(root):
    import json,numpy as np
    from . import tabletop_dual_pinch_builder as builder
    from .linear_coupling import expand
    pair=builder.load_pair_proposal()
    result=[dict(nodes=pair["rod_seed"]["proposed_nodes_m"],
        joints=pair["reset_seed"]["joint_positions"],roots=pair["reset_seed"]["reset_root_pose_w"])]
    cfg=json.loads((Path(root)/"configs/dh116_training/gpu_teacher_style_seeds_v1.json").read_text())
    spec=cfg["rod_seed_nodes"]
    for mode in MODES[1:]:
        entry=cfg["styles"][mode]
        nodes=np.zeros((41,3),dtype=np.float32)
        nodes[:,0]=spec["x_m"];nodes[:,1]=np.linspace(spec["y_min_m"],spec["y_max_m"],41)
        nodes[:,2]=spec["z_m"]
        result.append(dict(nodes=nodes.tolist(),
            joints=[expand(entry[s]["active_q_rad"]).tolist() for s in ("left","right")],
            roots=[entry[s]["base_pose_wxyz"] for s in ("left","right")]))
    return result


def compile_runtime(source,expected_sha):
    if hashlib.sha256(source.encode()).hexdigest()!=expected_sha:
        raise ValueError("Teacher source changed since preview configuration was frozen")
    tree=ast.parse(source)
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="main")
    start=next(i for i,n in enumerate(main.body) if isinstance(n,ast.ImportFrom)
        and n.module=="go2_pvcnn.control.dh116")
    stop=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=="init_std_val" for t in n.targets))
    body=ast.unparse(ast.Module(body=main.body[start:stop],type_ignores=[]))
    def replace(old,new,count):
        nonlocal body
        if body.count(old)!=count:raise ValueError("Teacher runtime layout differs: "+old[:70])
        body=body.replace(old,new)
    replace("nodes_batch = base_nodes[None, :, :] + env_origins_np[:, None, :]",
        "nodes_batch = np.asarray([s['nodes'] for s in preview_seeds], dtype=np.float32) + env_origins_np[:, None, :]",1)
    replace("device = 'cuda:0'","device = args.device",1)
    a=body.index("seed_q = torch.zeros((2, 17)")
    b=body.index("from go2_pvcnn.control.dh116.tabletop_material",a)
    body=body[:a]+"""seed_q = torch.zeros((2, N, 17), device=device, dtype=torch.float32)
for hi, hand in enumerate((left_hands,right_hands)):
    for env_id, entry in enumerate(preview_seeds):
        for j,name in enumerate(JOINTS):
            seed_q[hi,env_id,hand.joint_names.index(name)]=entry["joints"][hi][j]
seed_root_base=torch.tensor([[s["roots"][hi] for s in preview_seeds] for hi in range(2)],device=device,dtype=torch.float32)
""" + body[b:]
    replace("r_pose.unsqueeze(0).repeat(N, 1)","r_pose.clone()",2)
    replace("seed_q[hi:hi + 1].repeat(N, 1)","seed_q[hi].clone()",2)
    replace("float(np.median(base_nodes[:, 2]))",
        "torch.as_tensor(np.median(nodes_batch[:,:,2],axis=1),device=device)",1)
    replace("float(np.min(base_nodes[:, 2]))",
        "torch.as_tensor(np.min(nodes_batch[:,:,2],axis=1),device=device)",1)
    compiled=ast.parse("def create_runtime(args, app, preview_seeds):\n"+
        "\n".join("    "+line for line in body.splitlines())+"\n    return locals()\n")
    for node in ast.walk(compiled):
        if isinstance(node,ast.Attribute) and node.attr in ("_exit","backward"):
            raise ValueError("training or forced exit may not enter the viewer")
    return compiled


def create_runtime(root,args,app,source_sha):
    import importlib.util
    path=Path(root)/"scripts/train_dh116_gpu_teacher_ppo.py"
    source=path.read_text()
    code=compile_runtime(source,source_sha)
    spec=importlib.util.spec_from_file_location("_dh116_preview_teacher",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    namespace=dict(module.__dict__)
    exec(compile(code,str(path)+"[preview-only]","exec"),namespace)
    return SimpleNamespace(**namespace["create_runtime"](args,app,load_seeds(root)))


def export_physics_display(sim,interface):
    """Export the measured PhysX transforms, with zero simulation advancement."""
    before=(sim.current_time_step_index,float(sim.current_time))
    sim.physics_sim_view.update_articulations_kinematic()
    interface.update_transformations(True,True,True,False)
    after=(sim.current_time_step_index,float(sim.current_time))
    if after!=before:
        raise RuntimeError("display export changed the physics clock")
    return dict(physics_clock_unchanged=True,physics_step=before[0],
        physics_time_s=before[1],native_update_transformations=True)
