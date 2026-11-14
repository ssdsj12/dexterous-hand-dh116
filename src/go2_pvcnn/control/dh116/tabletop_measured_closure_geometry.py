"""Owned same-clock closure geometry; no stepping, native handles or qualification.

Construct after AppLauncher when used by a native worker. Only CPU source
geometry/SDF arrays are retained. capture borrows SDK objects for one read;
validate replays a saved owned packet without SDK access.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import numpy as np

from .tabletop_deformable_bimanual_step import _capture,_owned,_clock,DT,SIDES
from .tabletop_deformable_r0_replay import encode_owned,decode_owned
from .tabletop_deformable_physical_exchange import FlexiblePhysicalExchange
from .tabletop_palm_mount import CONTRACT as WRIST
from .tabletop_joint_target_readback import WRIST_LINEAR,WRIST_ANGULAR
from .linear_coupling import ACTIVE,JOINTS
from .tabletop_deformable_wrist_projection import CLEARANCE_M

VERSION="dh116-measured-closure-geometry-v1"
CLOCK_FIELDS=("run_uuid","env_id","episode_uid","reset_id","rod_uid","physics_step",
              "rod_tick","timestamp_s","solver_time_s","clock_valid")
REQUIRED=("finger14_link","finger23_link")
NUMERIC_GUARD=2e-6


def _bytes(value):
    return json.dumps(encode_owned(value),sort_keys=True,separators=(",",":"),allow_nan=False).encode()


def _array(value,shape):
    out=_owned(value)
    if out.shape!=shape or out.dtype.kind!="f" or not np.isfinite(out).all():
        raise ValueError("complete finite measured array required: "+str(shape))
    return out


def required_sdf_valid(row):
    value=row["minimum_gap_m"];gradient=row["gradient_norm_at_minimum"]
    return bool(value is not None and np.isfinite(value) and gradient is not None
        and np.isfinite(gradient) and gradient>0 and len(row["in_lattice_nodes"])>0)


def rod_support_lower(vertices,nodes):
    """Dual-feasible separation from ALL actual nodes, hence their whole polyline."""
    a=np.arange(64)*2*np.pi/64
    n=np.stack((np.cos(a),np.zeros(64),np.sin(a)),axis=1)
    n/=np.linalg.norm(n,axis=1)[:,None]+1e-12
    distance=max(0.,float(((vertices@n.T).min(0)-(nodes@n.T).max(0)).max()))
    return distance-.005-NUMERIC_GUARD


def _identity(exchange):
    identity={k:deepcopy(exchange["rod"][k]) for k in CLOCK_FIELDS}
    identity.update(policy_step=exchange["policy_step"],substep=exchange["substep"])
    _clock(lambda:[identity],(identity["env_id"],))
    if (identity["physics_step"]!=exchange["physics_step"]
            or exchange["action_readback"]["post"]["identity"]!=[identity]):
        raise ValueError("same completed action/rod identity required")
    return identity


class MeasuredClosureGeometry:
    def __init__(self,project):
        # Import the USD-backed loader only inside this post-App constructor.
        from scripts.audit_dh116_pinch_initial_pose import load_geometry,pose_matrix,transform,digest
        from scripts.prepare_dh116_pinch_closure_margin import sample_contact
        from scripts.solve_dh116_tripod_sdf_constrained_direction import grids_for,SDF_DIR,SDF_SHA,checked_json
        from .tabletop_dual_pinch_builder import load_pair_proposal
        project=Path(project).resolve();pair=load_pair_proposal();summary=checked_json(SDF_DIR/"result.json",SDF_SHA)
        self._pose_matrix,self._transform,self._sdf=pose_matrix,transform,sample_contact
        self._geometry={};self._grids={};self.sources={}
        for side in SIDES:
            g=load_geometry(project,side=side)
            expected=pair[side]["candidate"]["source_sha256"] if side=="left" else pair[side]["source_sha256"]
            if g["source_sha256"]!=expected or len(g["source_colliders"])!=17 or len(g["cooked_pads"])!=5:
                raise ValueError("original side-specific 17 source plus locked5 required")
            grids,binding=grids_for(g,summary)
            self._grids[side]=grids
            self._geometry[side]={kind:{n:np.array(row["vertices"],dtype=float,copy=True)
                for n,row in g[key].items()} for kind,key in (("source","source_colliders"),("cooked","cooked_pads"))}
            self.sources[side]=dict(source_sha256=deepcopy(g["source_sha256"]),sdf_binding=binding)
        files=("src/go2_pvcnn/control/dh116/tabletop_measured_closure_geometry.py",
            "scripts/audit_dh116_pinch_initial_pose.py","scripts/prepare_dh116_pinch_closure_margin.py",
            "scripts/solve_dh116_tripod_sdf_constrained_direction.py",
            "src/go2_pvcnn/control/dh116/tabletop_deformable_bimanual_step.py",
            "src/go2_pvcnn/control/dh116/tabletop_deformable_physical_exchange.py",
            "src/go2_pvcnn/control/dh116/tabletop_palm_mount.py")
        self.sources["implementation"]={n:digest(project/n) for n in files}
        self.sources["sdf_summary_sha256"]=SDF_SHA

    def capture(self,exchange,*,sim,bindings,read_clock):
        hand=view=binding=None
        try:
            expected=_identity(exchange);env=expected["env_id"]
            def current():
                c=deepcopy(read_clock())
                if any(c.get(k)!=expected[k] for k in CLOCK_FIELDS):
                    raise ValueError("SDK/rod clock changed during closure geometry capture")
                if (sim.current_time_step_index!=expected["physics_step"]
                        or not math.isclose(float(sim.current_time),expected["timestamp_s"],abs_tol=2e-7,rel_tol=0)):
                    raise ValueError("actual SDK boundary differs from completed rod step")
                return dict(c,policy_step=expected["policy_step"],substep=expected["substep"])
            current()
            snapshot=_capture(bindings,(env,),lambda:[current()])
            if _bytes(snapshot)!=_bytes(exchange["action_readback"]["post"]):
                raise ValueError("actual q/qd/targets changed after action post readback")
            raw=dict(identity=expected,joint_readback=snapshot,hands={},
                rod_nodes_m=_array(exchange["rod"]["node_positions"],(1,41,3))[0])
            if tuple(b.side for b in bindings)!=SIDES:raise ValueError("ordered distinct original hands required")
            for binding in bindings:
                hand=binding.hand;view=hand.root_physx_view;side=binding.side
                names=list(hand.body_names);paths=deepcopy(view.link_paths)
                prims=list(view.prim_paths)
                pads={n:exchange["rod"]["mesh_bindings"][side+"."+n] for n in self._grids[side]}
                containers={str(PurePosixPath(p["prim_path"]).parent) for p in pads.values()}
                if len(containers)!=1:raise ValueError("one independent authored hand container required")
                container=next(iter(containers))
                if (len(names)!=23 or len(set(names))!=23 or paths!=[[container+"/"+n for n in names]]
                        or prims!=[binding.env_paths[env]]
                        or not set(self._geometry[side]["source"])<=set(names)):
                    raise ValueError("actual 23-link identity or original source mapping differs")
                for n,p in pads.items():
                    b=self.sources[side]["sdf_binding"][n]
                    if (p["side"]!=side or p["body_name"]!=n or p["body_index"]!=names.index(n)
                            or p["prim_path"]!=paths[0][names.index(n)]
                            or p["sdf_sha256"]!=b["sha256"] or p["capture_sha256"]!=b["capture_sha256"]):
                        raise ValueError("same actual pad/SDF binding required")
                poses=_array(view.get_link_transforms(),(1,23,7))
                velocities=_array(view.get_link_velocities(),(1,23,6))
                if (names!=list(hand.body_names) or paths!=view.link_paths or prims!=list(view.prim_paths)):
                    raise ValueError("actual link identity changed during capture")
                raw["hands"][side]=dict(body_names=names,link_paths=paths,prim_paths=prims,
                    link_poses_xyzw=poses,link_velocities_world=velocities,
                    link_velocity_semantics="PhysX get_link_velocities world COM linear and angular velocity")
            current()
            return self._evaluate(raw,exchange)
        finally:
            self=exchange=sim=bindings=read_clock=hand=view=binding=None

    def _evaluate(self,raw,exchange):
        expected=_identity(exchange)
        if raw["identity"]!=expected or _bytes(raw["joint_readback"])!=_bytes(exchange["action_readback"]["post"]):
            raise ValueError("raw measured state or identity differs from completed exchange")
        nodes=_array(raw["rod_nodes_m"],(41,3))
        if not np.array_equal(nodes,_array(exchange["rod"]["node_positions"],(1,41,3))[0]):
            raise ValueError("geometry rod nodes differ from actual same-clock row")
        reasons=[];sides={}
        original=FlexiblePhysicalExchange._guard_failure(None,raw["joint_readback"]["hands"],
            exchange["rod"]["hand_wrenches_com_world"])
        if original:reasons.append("original:"+original)
        projection=exchange["wrist_projection"]
        if (bool(np.asarray(projection["safe_z_clipped"]).any())
                or np.any(abs(_owned(projection["palm_offsets_m"]))>1e-12)
                or np.any(abs(_owned(projection["projected_angles"]))>1e-12)):
            reasons.append("unexpected_wrist_projection")
        targets=_array(projection["projected_root_poses"],(1,2,7))
        for hi,side in enumerate(SIDES):
            data=raw["hands"][side];names=data["body_names"]
            if len(names)!=23 or len(set(names))!=23 or "base_link" not in names:
                raise ValueError("23 unique actual body names required")
            poses=_array(data["link_poses_xyzw"],(1,23,7))[0]
            vel=_array(data["link_velocities_world"],(1,23,6))[0]
            if not np.allclose(np.linalg.norm(poses[:,3:],axis=1),1.,atol=1e-5,rtol=0):
                raise ValueError("actual unit XYZW link quaternions required")
            joint=raw["joint_readback"]["hands"][side]
            if (len(joint["joint_names"])!=17 or len(set(joint["joint_names"]))!=17
                    or set(joint["joint_names"])!=set(JOINTS+WRIST_LINEAR+WRIST_ANGULAR)
                    or joint["active_indices"]!=[joint["joint_names"].index(n) for n in ACTIVE]
                    or joint["mounted"] is not True):
                raise ValueError("actual complete mounted17 joint state required")
            for field in ("q","qd","backend_targets"):_array(joint[field],(1,17))
            container=str(PurePosixPath(exchange["rod"]["mesh_bindings"][side+".finger14_link"]["prim_path"]).parent)
            if (data["link_paths"]!=[[container+"/"+n for n in names]] or data["prim_paths"]!=joint["prim_paths"]
                    or not set(self._geometry[side]["source"])<=set(names)):
                raise ValueError("persisted actual link/source identity differs")
            for n in self._grids[side]:
                b=exchange["rod"]["mesh_bindings"][side+"."+n]
                binding=self.sources[side]["sdf_binding"][n]
                if (b["body_index"]!=names.index(n) or b["prim_path"]!=data["link_paths"][0][names.index(n)]
                        or b["side"]!=side or b["body_name"]!=n
                        or b["sdf_sha256"]!=binding["sha256"] or b["capture_sha256"]!=binding["capture_sha256"]):
                    raise ValueError("persisted actual pad identity differs")
            matrices={n:self._pose_matrix(np.r_[p[:3],p[6],p[3:6]]) for n,p in zip(names,poses)}
            table={};gaps={}
            for kind,shapes in self._geometry[side].items():
                for n,vertices in shapes.items():
                    v=self._transform(vertices,matrices[n]);key=kind+":"+n
                    table[key]=float(v[:,2].min()-.3)
                    if n not in REQUIRED:gaps[key]=rod_support_lower(v,nodes)
            sdf={n:self._sdf(self._grids[side][n],matrices[n],nodes) for n in REQUIRED}
            for k,gap in table.items():
                if gap<CLEARANCE_M:reasons.append(side+":table:"+k)
            for k,gap in gaps.items():
                if gap<0:reasons.append(side+":non_target_not_proven_clear:"+k)
            for n,row in sdf.items():
                if not required_sdf_valid(row):reasons.append(side+":sdf_coverage:"+n)
            base=names.index("base_link")
            tracking=float(np.linalg.norm(poses[base,:3]-targets[0,hi,:3]))
            speed=float(np.linalg.norm(vel[base,:3]))
            if tracking>WRIST["tracking_guard_m"] or speed>WRIST["linear_speed_guard_m_s"]:
                reasons.append(side+":wrist_transport")
            q=np.asarray(joint["q"])[0]
            if any(abs(q[joint["joint_names"].index(n)])>math.pi/2 for n in WRIST_ANGULAR):
                reasons.append(side+":wrist_angular_limit")
            sides[side]=dict(table_clearance_m=table,minimum_table_clearance_m=min(table.values()),
                non_target_surface_gap_lower_m=gaps,minimum_non_target_gap_lower_m=min(gaps.values()),
                required_pad_sdf=sdf,actual_base_pose_xyzw=poses[base].tolist(),
                wrist_tracking_error_m=tracking,wrist_world_com_linear_speed_m_s=speed)
        return dict(version=VERSION,raw=deepcopy(raw),raw_sha256=hashlib.sha256(_bytes(raw)).hexdigest(),sources=deepcopy(self.sources),sides=sides,
            passed=not reasons,failure_reasons=reasons,
            terminal_reason=None if not reasons else "closure_geometry_guard",
            table_floor_m=CLEARANCE_M,rod_radius_m=.005,non_target_numeric_guard_m=NUMERIC_GUARD,
            required_target_negative_gap_allowed=True,geometry_scope="actual_source17_and_locked5_not_live_nonpad_cooked",
            non_target_bound_scope="all41_measured_node_convex_support_covers_centerline_polyline",
            hand_table_contact_available=False,no_hand_table_contact_proven=False,
            new_sdk_steps=0,training_updates=0,native_qualification=False,teacher_admission=False)

    def validate(self,packet,exchange):
        """Lossless recomputation, including failure rows; never trust stored flags."""
        if packet.get("version")!=VERSION or packet.get("sources")!=self.sources:
            raise ValueError("exact measured geometry implementation/source identity required")
        rebuilt=self._evaluate(decode_owned(encode_owned(packet["raw"])),exchange)
        if _bytes(rebuilt)!=_bytes(packet):
            raise ValueError("saved measured closure geometry differs from lossless recomputation")
        return rebuilt
