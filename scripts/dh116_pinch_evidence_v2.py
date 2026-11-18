"""CPU-only raw evidence checks for the bounded single-left Pinch successor."""
import numpy as np

from go2_pvcnn.control.dh116.deformx_dh116_colliders import DISTAL_PAD_LINKS
from go2_pvcnn.control.dh116.tabletop_deformx_pair_capture import validate_pair_capture

DT=1/480
PAIR_CAPACITY=41*6*12
CONTACT_CAPACITY=4096
IDENTITY=("phase_step","physics_step","rod_tick","episode_uid","reset_id","env_id","rod_uid","timestamp_s")


def own(value):
    if hasattr(value,"detach"):
        value=value.detach().cpu().numpy()
    return np.array(value,copy=True)


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [serial(v) for v in value]
    return value


def phase_pair_record(packet,row):
    raw=serial(packet)
    raw["indices_shape"]=list(np.asarray(packet["indices"]).shape)
    raw["vectors_shape"]=list(np.asarray(packet["vectors"]).shape)
    record={**{k:row[k] for k in IDENTITY},"raw":raw}
    record["pairs"]=validate_phase_pair_record(record,row)["pairs"]
    return record


def validate_phase_pair_record(record,row):
    if any(record.get(k)!=row[k] for k in IDENTITY):
        raise ValueError("pair clock/episode differs")
    raw=dict(record["raw"])
    attempted=raw["attempted"]
    if (raw["mesh_count"]!=6 or raw["substeps"]!=12 or raw["capacity"]!=PAIR_CAPACITY
            or type(attempted) is not int or not 0<=attempted<=PAIR_CAPACITY
            or raw.pop("indices_shape")!=[attempted,3] or raw.pop("vectors_shape")!=[attempted,12]):
        raise ValueError("pair shape/count differs from frozen six-mesh scope")
    # JSON cannot retain empty ndarray rank/dtype; use its explicitly recorded shape.
    ids=np.asarray(raw["indices"])
    vectors=np.asarray(raw["vectors"],dtype=float)
    if attempted==0:
        if ids.shape!=(0,) or vectors.shape!=(0,):
            raise ValueError("empty packet must retain its exact serialized shape metadata")
        ids=np.empty((0,3),dtype=np.int64);vectors=np.empty((0,12),dtype=float)
    raw["indices"],raw["vectors"]=ids,vectors
    node=np.asarray(raw["node_forces_on_rod_world_n"])
    mesh=np.asarray(raw["mesh_wrenches_on_mesh_world"])
    names=[*DISTAL_PAD_LINKS,"table"]
    expected=np.asarray([row["rod_mesh_wrenches_w"][n] for n in names])
    if (node.shape!=(41,3) or mesh.shape!=(6,6)
            or not np.allclose(node,row["rod_node_contact_forces_w"],atol=3e-4,rtol=5e-5)
            or not np.allclose(mesh,expected,atol=3e-4,rtol=5e-5)):
        raise ValueError("atomic pair packet and same-step aggregate force differ")
    result=validate_pair_capture(raw,dt_s=DT,node_forces=node,mesh_wrenches=mesh)
    for pair in result["pairs"]:
        pair["mesh_name"]=names[pair["mesh_index"]]
    if "pairs" in record and record["pairs"]!=result["pairs"]:
        raise ValueError("stored parsed pairs differ from the independent raw calculation")
    return result


def read_table_contacts(view,row):
    if view.sensor_count!=17 or view.filter_count!=1 or view.max_contact_data_count<1:
        raise ValueError("actual seventeen sensors and one table filter required")
    arrays=[own(v) for v in view.get_contact_data(DT)]
    if len(arrays)!=6:
        raise ValueError("actual six-buffer PhysX contact API required")
    force,point,normal,separation,counts,starts=arrays
    capacity=int(view.max_contact_data_count)
    if (counts.shape!=(17,1) or starts.shape!=(17,1) or counts.dtype.kind not in "iu"
            or starts.dtype.kind not in "iu" or np.any(counts<0) or np.any(starts<0)
            or force.shape!=(capacity,1) or point.shape!=(capacity,3)
            or normal.shape!=(capacity,3) or separation.shape!=(capacity,1)):
        raise ValueError("PhysX contact count/buffer shape differs")
    total=int(counts.sum())
    if total>=capacity or np.any(starts[counts>0]+counts[counts>0]>=capacity):
        raise ValueError("PhysX contact buffer saturated or exceeded")
    records=[];seen=set()
    for sensor in range(17):
        start,count=int(starts[sensor,0]),int(counts[sensor,0])
        if start>capacity:
            raise ValueError("invalid contact buffer index")
        for i in range(start,start+count):
            if i in seen:
                raise ValueError("overlapping PhysX contact buffer ranges")
            seen.add(i)
            if not all(np.isfinite(v[i]).all() for v in (force,point,normal,separation)):
                raise ValueError("nonfinite measured PhysX contact data")
            if not np.isclose(np.linalg.norm(normal[i]),1,atol=2e-5,rtol=0):
                raise ValueError("invalid measured contact normal")
            records.append(dict(sensor_index=sensor,buffer_index=i,normal_force_n=float(force[i,0]),
                point_world_m=point[i].tolist(),normal_world=normal[i].tolist(),
                separation_m=float(separation[i,0])))
    return {**{k:row[k] for k in IDENTITY},"source":"RigidContactView.get_contact_data",
            "sensor_count":17,"filter_count":1,"capacity":capacity,"count_buffer":counts.tolist(),
            "start_indices":starts.tolist(),"total_contacts":total,"contacts":records,
            "contact_report_enabled":True}


def validate_table_contacts(record,row):
    if (any(record.get(k)!=row[k] for k in IDENTITY)
            or record.get("source")!="RigidContactView.get_contact_data"
            or record.get("sensor_count")!=17 or record.get("filter_count")!=1
            or record.get("contact_report_enabled") is not True):
        raise ValueError("PhysX contact source/clock/identity differs")
    counts=np.asarray(record["count_buffer"]);starts=np.asarray(record["start_indices"])
    capacity=record["capacity"]
    if (type(capacity) is not int or capacity<1 or counts.shape!=(17,1) or starts.shape!=(17,1)
            or counts.dtype.kind not in "iu" or starts.dtype.kind not in "iu"
            or np.any(counts<0) or np.any(starts<0) or np.any(starts>capacity)):
        raise ValueError("PhysX contact count/index metadata differs")
    total=int(counts.sum())
    if (total>=capacity or record["total_contacts"]!=total or len(record["contacts"])!=total
            or np.any(starts[counts>0]+counts[counts>0]>=capacity)):
        raise ValueError("incomplete/saturated PhysX contact packet")
    expected={(s,i) for s in range(17) for i in range(int(starts[s,0]),int(starts[s,0]+counts[s,0]))}
    observed={(p["sensor_index"],p["buffer_index"]) for p in record["contacts"]}
    if expected!=observed or len(observed)!=total:
        raise ValueError("PhysX contact records differ from count/index buffers")
    return total


def require_quiet(pair_record,contact_record,row):
    pair=validate_phase_pair_record(pair_record,row)
    count=validate_table_contacts(contact_record,row)
    if count:
        raise ValueError("unexpected measured hand-table contact")
    if row["phase_step"]<=48 and any(p["mesh_index"]<5 for p in pair["pairs"]):
        raise ValueError("quiet window contains measured hand-rod pairs")
    return True


def validate_action_targets(rows,plan,cfg):
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,JOINTS,MIMIC,active_limits
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import candidate_action,integrate_pinch_action
    from go2_pvcnn.control.dh116.tabletop_task import joint_safety_failure
    from scripts.verify_dh116_deformx_rigid_bridge import wrench_at_com
    names=plan["joint_names"]
    if len(names)!=11 or set(names)!=set(JOINTS):
        raise ValueError("actual original joint names required")
    active=[names.index(n) for n in ACTIVE]
    target=np.asarray(cfg["initial_pose"]["active_q_rad"],dtype=float)
    if not np.allclose(np.asarray(rows[0]["pre_state"]["q"])[active],target,atol=1e-5,rtol=0):
        raise ValueError("first measured joint state differs from corrected reset")
    limits=np.stack(active_limits(),axis=-1)
    for step,row in enumerate(rows,1):
        action=candidate_action(cfg,step)
        if not np.array_equal(row["canonical_action24"],action):
            raise ValueError("measured action differs from frozen quiet/move phases")
        before=row["pre_state"];after=row["post_state"]
        target=integrate_pinch_action(target,np.asarray(before["q"])[active],action,side="left",dt=DT)
        if not np.allclose(np.asarray(row["sdk_joint_target_rad"])[active],target,atol=1e-7,rtol=0):
            raise ValueError("actual SDK target differs from original action kernel")
        q=np.asarray(after["q"])
        mimic=np.asarray([q[names.index(f)]-(ratio*q[names.index(m)]+off) for f,m,ratio,off in MIMIC])
        if joint_safety_failure(q[active],limits,mimic) is not None:
            raise ValueError("measured joint/mimic safety failed")
        reaction=np.stack([wrench_at_com(row["rod_mesh_wrenches_w"][name],
            after["link_pos"][i],after["com_pos"][i]) for i,name in enumerate(DISTAL_PAD_LINKS)])
        if not np.allclose(reaction,row["rod_on_hand_com_w"],atol=2e-5,rtol=2e-5):
            raise ValueError("actual mesh-origin reaction was not shifted to named body COM")
