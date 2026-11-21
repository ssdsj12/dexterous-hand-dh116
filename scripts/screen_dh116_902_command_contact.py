"""Separately versioned, full-substep command proof for the measured 902 Tripod boundary.

Predicts ideal submitted-target tracking only. It never qualifies native contact.
USD/FK loading is confined to prepare(), which the launch entry runs in a CPU child.
"""
from pathlib import Path
import json
import numpy as np

VERSION = "dh116-902-command-contact-v2"
SCOPE = "ideal_float32_command_tracking_against_archived_envelope"
SIDES = ("left", "right")
PADS = ("finger14_link", "finger23_link", "finger33_link")


def checked_request(request, *, grasp, sdk, outer_sha):
    from scripts.audit_dh116_certified_segment_distance import ENVELOPE_REFINEMENT_VERSION
    q = request.get("target", {}).get("active_q_rad")
    pose = request.get("target", {}).get("pose_wxyz")
    relief = request.get("temporary_relief_rad", {})
    if (request.get("version") != VERSION or grasp != "tripod" or request.get("grasp") != grasp
            or request.get("contact_side") != "right" or sdk != 408 or request.get("source_sdk") != sdk
            or request.get("source_outer_sha256") != outer_sha
            or type(request.get("control_windows")) is not int or request["control_windows"] != 66
            or type(request.get("hold_windows")) is not int or request["hold_windows"] != 0
            or request.get("max_penetration_m") != .00015 or request.get("support_reserve_m") != .001
            or request.get("envelope_refinement") != ENVELOPE_REFINEMENT_VERSION
            or type(request.get("index_delay_windows")) is not int
            or not 0 <= request["index_delay_windows"] < 66
            or not isinstance(relief, dict) or set(relief) != {"8", "9"}
            or any(type(x) not in (float,int) or not 0 <= x <= .005 for x in relief.values())
            or np.shape(q) != (6,) or np.shape(pose) != (7,)
            or not np.isfinite(q).all() or not np.isfinite(pose).all()
            or abs(np.linalg.norm(pose[3:])-1) > 1e-8):
        raise ValueError("bounded source-bound Tripod/right command-contact v2 request required")
    from go2_pvcnn.control.dh116.linear_coupling import expand
    expand(q)


def apply_schedule(base, request):
    from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
    a = np.asarray(base, dtype=float).copy()
    held = list(range(6)) + list(range(12,15)) + list(range(18,21))
    if a.shape != (66,24) or not np.isfinite(a).all() or np.any(a[:,held] != 0.):
        raise ValueError("finite exact 66-window commands with unchanged held hand required")
    delay = request["index_delay_windows"]
    if type(delay) is not int or not 0 <= delay < 66:
        raise ValueError("bounded delay required")
    original_sum = a.sum(0)
    # Re-distribute the original constant index closing command, not the target.
    if not np.all(a[:,8] == a[0,8]):
        raise ValueError("constant original index approach command required")
    a[:delay,8] = 0.
    a[delay:,8] *= 66/(66-delay)
    timeline = np.arange(67)/66
    for slot, amplitude in request["temporary_relief_rad"].items():
        offset = np.interp(timeline,[0.,.35,.65,.93,1.],[0.,0.,-amplitude,-amplitude,0.])
        a[:,int(slot)] += np.diff(offset)/(8/480*ACTION_CONTRACT["joint_rate_rad_s"])
    if not np.isfinite(a).all() or np.max(abs(a)) > 1. or not np.allclose(a.sum(0),original_sum,atol=1e-12,rtol=0):
        raise ValueError("original action range and endpoint integral required")
    return a


def quantized_window_commands(actions, initial_command):
    """Match each original joint waypoint after the existing float32 SDK writes.

    Only compiles the right-hand command list. It neither measures nor predicts
    physical tracking; the full command kernel proof still checks its guards.
    """
    from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
    a=np.asarray(actions,dtype=float);q=np.asarray(initial_command,dtype=float).copy()
    if (a.shape!=(66,24) or q.shape!=(6,) or not np.isfinite(a).all()
            or not np.isfinite(q).all() or np.max(abs(a))>1.):
        raise ValueError("finite bounded original command sequence required")
    result=a.copy();desired=q.copy()
    scale=8/480*ACTION_CONTRACT["joint_rate_rad_s"]
    for i in range(len(a)):
        desired+=a[i,6:12]*scale
        corrected=(desired-q)/scale
        # Numerical compilation only: never compensate physical tracking error.
        if np.max(abs(corrected-a[i,6:12]))*scale>1e-6 or np.max(abs(corrected))>1.:
            raise ValueError("float32 waypoint correction exceeds numerical budget")
        result[i,6:12]=corrected
        for _ in range(8):
            q=(q+corrected*scale/8).astype(np.float32).astype(float)
        if np.max(abs(q-desired))>1e-6:
            raise ValueError("float32 waypoint remains unresolved")
    return result


def compile_actions(config, row, request):
    from scripts.screen_dh116_902_flexible_contact import stage_actions
    scheduled=apply_schedule(stage_actions(config,row,request),request)
    h=row["action_readback"]["post"]["hands"]["right"]
    initial=h["backend_targets"][0,h["active_indices"]]
    return quantized_window_commands(scheduled,initial)


def support_bound(vertices, low, high, *, reserve_m):
    from scripts.certify_dh116_tripod_distal_route import support_clearance_lower
    from scripts.audit_dh116_pinch_initial_pose import box_relation
    low,high=np.asarray(low),np.asarray(high)
    fast=float(support_clearance_lower(vertices,low,high))
    sat=None
    if fast < reserve_m:
        sat=box_relation(vertices,(low+high)/2,high-low)["max_separating_axis_gap_m"]-1e-12
    return dict(legacy_lower_m=fast,sat_lower_m=sat,lower_m=fast if sat is None else max(fast,sat))


def route_passed(proof, request):
    try:
        rows=proof["samples"];count=request["control_windows"]*8
        if (proof["version"]!=VERSION or proof["grasp"]!="tripod" or proof["contact_side"]!="right"
                or proof["scope"]!=SCOPE or proof["control_windows"]!=request["control_windows"]
                or proof["before_sdk"]!=408 or proof["after_sdk"]!=408+count or len(rows)!=count
                or proof["native_contact_verified"] is not False or proof["teacher_admission"] is not False):
            return False
        for i,row in enumerate(rows,1):
            if row["ordinal"]!=i or set(row["hand_rows"])!=set(SIDES): return False
            for side,hand in row["hand_rows"].items():
                values=[hand[k] for k in ("noncontact_rod_clearance_m","support_clearance_m","table_clearance_m")]
                if (hand["shape_count"]!=22 or not np.isfinite(values).all() or values[0]<=.00025
                        or values[1]<request["support_reserve_m"] or values[2]<.0021):
                    return False
                if side=="right":
                    gaps=hand["pad_gaps_m"]; ids=hand["node_ids"]
                    if (hand["all_envelopes_certified"] is not True or len(gaps)!=3 or len(ids)!=3
                            or not np.isfinite(gaps).all() or min(gaps)<-request["max_penetration_m"]
                            or not np.isfinite(hand["contact_rod_clearance_m"])
                            or hand["contact_rod_clearance_m"] < -request["max_penetration_m"]
                            or any(type(k) is not int or not 0<=k<20 for k in ids)):
                        return False
        last=rows[-1]["hand_rows"]["right"]
        errors=proof["endpoint_error"]
        if (max(last["pad_gaps_m"])>=0. or len(last["opposition"])!=2
                or not np.isfinite(last["opposition"]).all() or max(last["opposition"])>=-.2
                or not np.isfinite(list(errors.values())).all()
                or not 0<=errors["position_m"]<=1e-6 or not 0<=errors["rotation_rad"]<=1e-6
                or not 0<=errors["joint_rad"]<=1e-5):
            return False
        return True
    except (KeyError,TypeError,ValueError,IndexError):
        return False


def stations(proof):
    return {"right."+pad:sorted({row["hand_rows"]["right"]["node_ids"][j] for row in proof["samples"]})
            for j,pad in enumerate(PADS)}


def dependency_files():
    from scripts import verify_dh116_902_flexible_reset as reset
    from scripts.screen_dh116_902_flexible_contact import contact_dependency_files
    return (*contact_dependency_files(),Path(__file__),Path(__file__).with_name("screen_dh116_902_command_path.py"),
            *(reset.ROOT/name for name in (
                "src/go2_pvcnn/control/dh116/tabletop_command_kernels.py",
                "src/go2_pvcnn/control/dh116/tabletop_palm_mount.py",
                "src/go2_pvcnn/control/dh116/tabletop_deformable_wrist_projection.py",
                "src/go2_pvcnn/control/dh116/tabletop_surface_contract.py",
                "src/go2_pvcnn/control/dh116/linear_coupling.py",
                "src/go2_pvcnn/control/dh116/hand_point_kinematics.py",
                "src/go2_pvcnn/control/dh116/usd_surface.py",
                "src/go2_pvcnn/control/dh116/usd_contract.py")))


def _measure_hand(side, context, root, q, rod_samples, tube, boxes, grids, table_top, request):
    from scripts.prepare_dh116_pinch_closure_margin import fk_poses,sample_grid
    from scripts.screen_dh116_tripod_supported_reset import separation
    from scripts.audit_dh116_certified_segment_distance import rod_envelope_clearance
    model,corrections=context
    fk=fk_poses(model,root,q)
    for name,matrix in corrections.items():
        fk[name]=fk[name]@matrix
    world={group+":"+name:shape["vertices"]@fk[name][:3,:3].T+fk[name][:3,3]
           for group in ("source_colliders","cooked_pads") for name,shape in model[group].items()}
    fast={label:float(separation(v,*tube)) for label,v in world.items()}
    contact={k:v for k,v in fast.items() if side=="right" and k.split(":")[1] in PADS}
    refinement={}
    for name,value in list(contact.items()):
        if value < -request["max_penetration_m"]:
            result=rod_envelope_clearance(world[name],rod_samples,max_penetration_m=request["max_penetration_m"])
            refinement[name]=result
            contact[name]=result["lower_m"]
    supports=[dict(shape=name,support=i,**support_bound(v,low,high,reserve_m=request["support_reserve_m"]))
              for name,v in world.items() for i,(low,high) in enumerate(boxes)]
    limiting=min(supports,key=lambda x:x["lower_m"])
    report=dict(shape_count=len(world),
        noncontact_rod_clearance_m=min(value for name,value in fast.items() if name not in contact),
        support_clearance_m=limiting["lower_m"],support_witness=limiting,
        table_clearance_m=min(float(v[:,2].min()-table_top) for v in world.values()))
    if side=="right":
        gaps=[];ids=[];normals=[]
        for pad in PADS:
            transform=fk[pad];local=(rod_samples[-1]-transform[:3,3])@transform[:3,:3]
            phi,grad,valid=sample_grid(grids["right."+pad],local,dtype=np.float64)
            if not valid.any(): raise ValueError("original pad SDF has no valid witness")
            i=int(np.nanargmin(phi));normal=transform[:3,:3]@grad[i]
            length=float(np.linalg.norm(normal))
            if length<=0 or not np.isfinite(length): raise ValueError("finite original pad normal required")
            ids.append(i);gaps.append(float(phi[i]-.005));normals.append(normal/length)
        report.update(contact_rod_clearance_m=min(contact.values()),pad_gaps_m=gaps,node_ids=ids,
            opposition=[float(normals[0]@n) for n in normals[1:]],refinement=refinement,
            all_envelopes_certified=all(x["passed"] for x in refinement.values()))
    return report


def prepare(receipt, request_path, *, expected_grasp=None):
    import time
    from scipy.spatial.transform import Rotation
    from scripts import verify_dh116_902_flexible_reset as reset
    from scripts.screen_dh116_902_flexible_contact import boundary
    from scripts.screen_dh116_902_command_path import command_step
    from scripts.audit_dh116_pinch_initial_pose import load_geometry
    from scripts.prepare_dh116_pinch_closure_margin import fk_poses
    from scripts.screen_dh116_tripod_supported_reset import archived_tube
    from scripts.diagnose_dh116_teacher_initialization import digest
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_scene_preparation as scene
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE
    start=time.monotonic()
    request=json.loads(request_path.read_bytes())
    grasp=request.get("grasp")
    if expected_grasp is not None and grasp!=expected_grasp: raise ValueError("same grasp required")
    raw,supports,files,sdk=boundary(receipt,grasp)
    checked_request(request,grasp=grasp,sdk=sdk,outer_sha=reset.sha(receipt))
    reference,config,config_files=reset.load_candidate(grasp)
    rows=[decode_owned(row) for row in raw];row=rows[-1]
    rod_samples=np.stack([v["rod"]["node_positions"][0] for v in rows])
    if rod_samples.shape!=(9,41,3): raise ValueError("complete nine-frame sealed rod envelope required")
    tube=archived_tube(rod_samples)
    boxes=[(np.array(b["center_m"])-np.array(b["size_m"])/2,np.array(b["center_m"])+np.array(b["size_m"])/2) for b in supports]
    table=reference["table"];table_top=table["center_m"][2]+table["size_m"][2]/2
    grids=scene.load_sdf_arrays(scene.prepare_scene(reset.ROOT))
    seed=np.array([[config["hands"][s]["pose_wxyz"] for s in SIDES]])
    poses=row["sdk_hand_geometry"]["poses"];hands=row["action_readback"]["post"]["hands"]
    roots=np.array([[np.r_[poses[s+".base_link"][0],poses[s+".base_link"][1][[3,0,1,2]]] for s in SIDES]])
    command=np.concatenate([hands[s]["backend_targets"][:,hands[s]["active_indices"]] for s in SIDES],1)
    q=np.concatenate([hands[s]["q"][:,hands[s]["active_indices"]] for s in SIDES],1)
    anchors=np.stack([hands[s]["mount_anchor_pose_w"] for s in SIDES],axis=1)
    palms=row["wrist_projection"]["palm_offsets_m"].copy();angles=row["wrist_projection"]["projected_angles"].copy()
    clearance=np.asarray(row["sdk_hand_geometry"]["table_clearance_m"]).reshape(1,2).copy()
    contexts={};model_sources={}
    for i,side in enumerate(SIDES):
        if [hands[side]["joint_names"][k] for k in hands[side]["active_indices"]]!=list(ACTIVE):
            raise ValueError("original active joint order required")
        model=load_geometry(reset.ROOT,side=side)
        if model["usd"]["sha256"]!=reference["hand_usd_sha256"][side]: raise ValueError("original side-specific USD required")
        actual={}
        for key,(xyz,quat) in poses.items():
            if key.startswith(side+"."):
                matrix=np.eye(4);matrix[:3,:3]=Rotation.from_quat(quat).as_matrix();matrix[:3,3]=xyz
                actual[key.split(".")[1]]=matrix
        old_fk=fk_poses(model,roots[0,i],q[0,6*i:6*i+6])
        corrections={name:np.linalg.solve(old_fk[name],matrix) for name,matrix in actual.items()}
        contexts[side]=(model,corrections);model_sources[side]=model["source_sha256"]
    actions=compile_actions(config,row,request)
    samples=[]
    for ordinal in range(len(actions)*8):
        if time.monotonic()-start>180: raise RuntimeError("bounded CPU command proof exceeded 180 seconds")
        out=command_step(action=actions[ordinal//8:ordinal//8+1],backend_targets=command,measured_q=q,
            seed=seed,measured_roots=roots,clearance=clearance,palms=palms,angles=angles,mount_anchor_poses=anchors)
        command=out["submitted_finger_targets"].astype(float);q=command.copy()
        for i in range(2):
            anchor=anchors[0,i];w=out["submitted_wrist_targets"][0,i].astype(float)
            rot=Rotation.from_quat(anchor[[4,5,6,3]])
            roots[0,i,:3]=anchor[:3]+rot.apply(w[:3])
            roots[0,i,3:]=(rot*Rotation.from_euler("XYZ",w[3:])).as_quat()[[3,0,1,2]]
        palms=out["projection"]["palm_offsets_m"];angles=out["projection"]["projected_angles"]
        reports={side:_measure_hand(side,contexts[side],roots[0,i],q[0,6*i:6*i+6],
                 rod_samples,tube,boxes,grids,table_top,request) for i,side in enumerate(SIDES)}
        clearance=np.array([[reports[s]["table_clearance_m"] for s in SIDES]])
        samples.append(dict(ordinal=ordinal+1,hand_rows=reports))
    target=request["target"];target_pose=np.asarray(target["pose_wxyz"])
    rotation=Rotation.from_quat(roots[0,1,[4,5,6,3]]).inv()*Rotation.from_quat(target_pose[[4,5,6,3]])
    proof=dict(version=VERSION,scope=SCOPE,grasp=grasp,contact_side="right",
        before_sdk=sdk,after_sdk=sdk+8*len(actions),control_windows=len(actions),
        request_sha256=reset.sha(request_path),outer_sha256=reset.sha(receipt),
        archive_digest=digest(raw[-1]),rod_samples_sha256=digest(rod_samples.tolist()),
        model_sources=model_sources,source_sha256={str(p.relative_to(reset.ROOT)):reset.sha(p) for p in dependency_files()},
        samples=samples,action_sha256=digest(actions.tolist()),
        endpoint_error=dict(position_m=float(np.linalg.norm(roots[0,1,:3]-target_pose[:3])),
            rotation_rad=float(rotation.magnitude()),joint_rad=float(np.max(abs(q[0,6:]-target["active_q_rad"])))),
        native_contact_verified=False,teacher_admission=False)
    proof["all_passed"]=route_passed(proof,request)
    proof["stations"]=stations(proof)
    model=contexts["right"][0]
    proof["continuous_cooked_hulls"]={"right."+pad:model["cooked_pads"][pad]["vertices"].tolist() for pad in PADS}
    contact=_contact(proof,request,raw[-1])
    return proof,actions,contact,(*files,*config_files,request_path,*dependency_files())


def _contact(proof, request, archive):
    return dict(entry_contract=VERSION,grasp="tripod",contact_side="right",boundary_sdk=408,
        archive=archive,archive_digest=proof["archive_digest"],stations=proof["stations"],
        max_pad_penetration_m=request["max_penetration_m"],
        continuous_cooked_hulls=proof["continuous_cooked_hulls"],
        envelope_refinement=request["envelope_refinement"],command_proof=proof,command_request=request)


def independent_entry(task, contact):
    from scripts.screen_dh116_902_flexible_contact import independent_entry as legacy
    proof=contact.get("command_proof",{});request=contact.get("command_request",{})
    checked_request(request,grasp=task,sdk=contact.get("boundary_sdk"),outer_sha=proof.get("outer_sha256"))
    if (contact.get("entry_contract")!=VERSION or not route_passed(proof,request)
            or proof.get("all_passed") is not True or stations(proof)!=contact.get("stations")
            or proof.get("continuous_cooked_hulls")!=contact.get("continuous_cooked_hulls")
            or proof.get("archive_digest")!=contact.get("archive_digest")
            or contact.get("envelope_refinement")!=request["envelope_refinement"]):
        raise ValueError("complete source-bound actual-command contact proof required")
    old=dict(contact,entry_contract="dh116-902-distal-contact-v1",envelope_refinement=None)
    return legacy(task,old)


def load_stage(receipt, request_path, proof_path, grasp, *, before_sdk):
    from scripts import verify_dh116_902_flexible_reset as reset
    from scripts.screen_dh116_902_flexible_contact import boundary
    from scripts.diagnose_dh116_teacher_initialization import digest
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
    request=json.loads(request_path.read_bytes());proof=json.loads(proof_path.read_bytes())
    raw,_,files,sdk=boundary(receipt,grasp)
    checked_request(request,grasp=grasp,sdk=sdk,outer_sha=reset.sha(receipt))
    _,config,config_files=reset.load_candidate(grasp)
    actions=compile_actions(config,decode_owned(raw[-1]),request)
    expected_sources={str(p.relative_to(reset.ROOT)):reset.sha(p) for p in dependency_files()}
    if (sdk!=before_sdk or proof.get("request_sha256")!=reset.sha(request_path)
            or proof.get("outer_sha256")!=reset.sha(receipt) or proof.get("archive_digest")!=digest(raw[-1])
            or proof.get("source_sha256")!=expected_sources or proof.get("action_sha256")!=digest(actions.tolist())):
        raise ValueError("frozen v2 request, source and original 24D commands required")
    contact=_contact(proof,request,raw[-1]);independent_entry(grasp,contact)
    return actions,contact,(*files,*config_files,request_path,proof_path,*dependency_files())
