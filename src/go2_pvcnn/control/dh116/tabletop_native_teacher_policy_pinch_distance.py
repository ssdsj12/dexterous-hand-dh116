"""Independent policy-owned Pinch geometry/distance, without a scripted recipe.

Only SDK link transforms, fixed 41 material nodes and locked pad SDFs are used.
The legacy schedule validators and their frozen step/action bounds are unchanged.
A delivered row is eligible only AFTER the real history owner records its exact
request/targets/token/window/hash. CPU fixtures remain explicitly CPU scoped.
"""
from copy import deepcopy
import hashlib
import math
from pathlib import Path,PurePosixPath
import numpy as np

from .tabletop_measured_closure_geometry import (
    MeasuredClosureGeometry,_array,_bytes,_identity,CLOCK_FIELDS,REQUIRED,
    required_sdf_valid,NUMERIC_GUARD,rod_support_lower)
from .tabletop_r0_measured_approach import MeasuredPinchApproach
from .tabletop_deformable_physical_exchange import FlexiblePhysicalExchange
from .tabletop_deformable_wrist_projection import SOURCE as PROJECTION_SOURCE,CLEARANCE_M
from .tabletop_palm_mount import CONTRACT as WRIST
from .tabletop_joint_target_readback import WRIST_LINEAR,WRIST_ANGULAR
from .linear_coupling import ACTIVE,JOINTS
from .tabletop_deformable_r0_replay import _digest,decode_owned,encode_owned
from .tabletop_training_reward import SourceFailure

VERSION="dh116-policy-owned-pinch-distance-v1"
GEOMETRY_VERSION="dh116-policy-owned-pinch-geometry-v1"


def _require(value,message):
    if not value:raise SourceFailure(message)


def _radius(exchange):
    from scripts.replay_dh116_native_parameter_r0 import check_radius_packet
    try:
        from . import tabletop_mesh_observation_source as source
        from .tabletop_native_teacher_profile import policy_source_basis
        packet=exchange["material_director"]
        _require(packet.get("version")==source.PARAMETER_VERSION and packet.get("sources")==policy_source_basis(),
            "current policy parameter reader/source inventory differs")
        return check_radius_packet(packet,exchange["rod"])
    except (ValueError,KeyError,TypeError) as exc:raise SourceFailure("actual same-clock radius: "+str(exc)) from exc


class PolicyOwnedPinchGeometry(MeasuredClosureGeometry):
    """Reuse only the locked asset loader and same-clock SDK capture from v1.

    _evaluate is a separate policy contract. No frozen schedule, absolute step
    origin, zero wrist request or diagnostic SDK limit is inherited.
    """
    REQUIRED_PADS = REQUIRED
    ROUTE_COUNT = 5
    VERSION = GEOMETRY_VERSION

    def __init__(self,project,*,scope):
        if scope not in ("cpu_fixture","owner_source_unqualified"):raise ValueError("explicit unqualified source scope required")
        super().__init__(project)
        self.scope=scope;self.project=Path(project).resolve()
        paths=(Path(__file__).name,"tabletop_polyline_support_certificate.py")
        self.sources["policy_owned"]={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in paths}

    def capture(self,*args,**kwargs):
        try:return super().capture(*args,**kwargs)
        except SourceFailure:raise
        except (ValueError,KeyError,TypeError,IndexError) as exc:
            raise SourceFailure("policy-owned SDK source capture: "+str(exc)) from exc

    def _radius(self, exchange):
        return _radius(exchange)

    def _expected_radius(self):
        import json
        return json.loads((self.project/"configs/dh116_tasks/flexible_pinch_candidate_v1.json").read_bytes())["rod"]["radius_m"]

    def _pad_samples(self,side,matrices,nodes,*,radius):
        return {n:self._sdf(self._grids[side][n],matrices[n],nodes,radius=radius) for n in self.REQUIRED_PADS}

    def _shape_metrics(self,vertices,matrix,*,side,key,nodes):
        return self._transform(vertices,matrix),None,None

    def _seal_result(self,raw):
        return dict(raw=deepcopy(raw),raw_sha256=hashlib.sha256(_bytes(raw)).hexdigest(),
                    sources=deepcopy(self.sources))

    def _evaluate(self,raw,exchange):
        expected=_identity(exchange)
        _require(raw["identity"]==expected and self._same_owned(raw["joint_readback"],exchange["action_readback"]["post"]),
            "same-clock owned SDK pose/joint readback required")
        nodes=_array(raw["rod_nodes_m"],(41,3))
        _array(exchange["rod"]["velocities"],(1,41,3))
        _array(exchange["rod"]["distributed_contact"],(1,41,3))
        for side in ("left","right"):
            for field in ("forces","torques"):
                _array(exchange["rod"]["hand_wrenches_com_world"][side][field],(1,self.ROUTE_COUNT,3))
        _require(np.array_equal(nodes,_array(exchange["rod"]["node_positions"],(1,41,3))[0]),"actual fixed material41 nodes differ")
        packet_scope=exchange["material_director"].get("source_scope")
        _require((packet_scope=="cpu_fixture")==(self.scope=="cpu_fixture"),"parameter/source fixture scope differs")
        radius=self._radius(exchange)["value"]
        # Existing conservative support certificate is defined at this unchanged
        # locked task radius. An incompatible radius is unknown, never relabelled.
        from .tabletop_polyline_support_certificate import certify,validate as validate_certificate,RADIUS
        expected_radius=self._expected_radius()
        _require(math.isclose(radius,expected_radius,rel_tol=2e-6,abs_tol=1e-9),"locked physical radius differs")
        projection=exchange["wrist_projection"];submission=exchange["action_readback"]["submission"]
        _require(projection["projection_source"]==submission["projection_source"]==PROJECTION_SOURCE,"actual wrist projection source differs")
        targets=_array(projection["projected_root_poses"],(1,2,7))
        angles=_array(projection["projected_angles"],(1,2,3))
        _require(np.array_equal(targets,_array(submission["projected_root_poses"],(1,2,7)))
            and np.array_equal(angles,_array(submission["projected_angles"],(1,2,3))),"actual projected/submitted SDK targets differ")
        action=_array(exchange["action_readback"]["request"]["action24"],(1,24))
        _require(np.all(abs(action)<=1),"finite normalized policy request24 required")
        reasons=[];source_errors=[];sides={};whole={}
        original=FlexiblePhysicalExchange._guard_failure(None,raw["joint_readback"]["hands"],exchange["rod"]["hand_wrenches_com_world"])
        if original:reasons.append("original:"+original)
        for hi,side in enumerate(("left","right")):
            data=raw["hands"][side];names=data["body_names"];joint=raw["joint_readback"]["hands"][side]
            _require(len(names)==23 and len(set(names))==23 and "base_link" in names,"23 unique actual SDK links required")
            poses=_array(data["link_poses_xyzw"],(1,23,7))[0];vel=_array(data["link_velocities_world"],(1,23,6))[0]
            _require(np.allclose(np.linalg.norm(poses[:,3:],axis=1),1.,atol=1e-5,rtol=0),"unit actual SDK XYZW poses required")
            _require(len(joint["joint_names"])==17 and set(joint["joint_names"])==set(JOINTS+WRIST_LINEAR+WRIST_ANGULAR)
                and joint["active_indices"]==[joint["joint_names"].index(n) for n in ACTIVE]
                and joint["mounted"] is True,"complete original mounted17 source required")
            for field in ("q","qd","backend_targets"):_array(joint[field],(1,17))
            container=str(PurePosixPath(exchange["rod"]["mesh_bindings"][side+".finger14_link"]["prim_path"]).parent)
            _require(data["link_paths"]==[[container+"/"+n for n in names]] and data["prim_paths"]==joint["prim_paths"]
                and set(self._geometry[side]["source"])<=set(names),"same actual side/link/source mapping required")
            for n in self._grids[side]:
                b=exchange["rod"]["mesh_bindings"][side+"."+n];locked=self.sources[side]["sdf_binding"][n]
                _require(b["body_index"]==names.index(n) and b["prim_path"]==data["link_paths"][0][names.index(n)]
                    and b["side"]==side and b["body_name"]==n and b["sdf_sha256"]==locked["sha256"]
                    and b["capture_sha256"]==locked["capture_sha256"],"locked actual pad/SDF source differs")
            matrices={n:self._pose_matrix(np.r_[p[:3],p[6],p[3:6]]) for n,p in zip(names,poses)}
            tables={};gaps={};vertices_all=[]
            for kind,shapes in self._geometry[side].items():
                for n,vertices in shapes.items():
                    key=kind+":"+n
                    v,table_hint,lower_hint=self._shape_metrics(vertices,matrices[n],side=side,key=key,nodes=nodes)
                    vertices_all.append(v)
                    tables[key]=float(v[:,2].min()-.3) if table_hint is None else table_hint
                    if n not in self.REQUIRED_PADS:
                        lower=(rod_support_lower(v,nodes) if lower_hint is None else lower_hint)+RADIUS-radius
                        if lower<0:
                            exact_vertices=v if lower_hint is None else self._transform(vertices,matrices[n])
                            certificate=certify(exact_vertices,nodes);validate_certificate(certificate,exact_vertices,nodes)
                            lower=certificate["lower_m"]+RADIUS-radius
                        gaps[key]=lower
            whole[side]=np.concatenate(vertices_all)
            sdf=self._pad_samples(side,matrices,nodes,radius=radius)
            for key,gap in tables.items():
                if gap<CLEARANCE_M:reasons.append(side+":table:"+key)
            for key,gap in gaps.items():
                if gap<0:reasons.append(side+":non_target_not_proven_clear:"+key)
            for n,pad in sdf.items():
                if not required_sdf_valid(pad):source_errors.append(side+":sdf_coverage:"+n)
            base=names.index("base_link");tracking=float(np.linalg.norm(poses[base,:3]-targets[0,hi,:3]));speed=float(np.linalg.norm(vel[base,:3]))
            if tracking>WRIST["tracking_guard_m"] or speed>WRIST["linear_speed_guard_m_s"]:reasons.append(side+":wrist_transport")
            q=np.asarray(joint["q"])[0]
            if any(abs(q[joint["joint_names"].index(n)])>math.pi/2 for n in WRIST_ANGULAR):reasons.append(side+":wrist_angular_limit")
            sides[side]=dict(table_clearance_m=tables,non_target_surface_gap_lower_m=gaps,
                required_pad_sdf=sdf,wrist_tracking_error_m=tracking,wrist_world_com_linear_speed_m_s=speed)
        # Pure separating-plane math over complete shape unions, no old recipe.
        left,right=whole["left"],whole["right"]
        separation=float(max(np.max(right.min(0)-left.max(0)),np.max(left.min(0)-right.max(0))))
        if separation<0:reasons.append("inter_hand_separation_not_proven")
        return dict(version=self.VERSION,source_scope=self.scope,**self._seal_result(raw),
            sides=sides,source_complete=not source_errors,source_errors=source_errors,
            passed=not reasons and not source_errors,failure_reasons=reasons,
            terminal_reason=None if not reasons else "policy_owned_source_geometry_guard",
            original_guard_reason=original,inter_hand_separation_lower_m=separation,rod_radius_m=radius,
            fixed_material_ids=list(range(41)),new_sdk_steps=0,native_qualification=False,teacher_admission=False)

    def validate(self,packet,exchange):
        _require(packet.get("version")==self.VERSION and packet.get("sources")==self.sources
            and packet.get("source_scope")==self.scope,"policy-owned geometry source/version/scope differs")
        rebuilt=self._evaluate(decode_owned(encode_owned(packet["raw"])),exchange)
        _require(_bytes(rebuilt)==_bytes(packet),"owned SDK pose/clock/source geometry recomputation differs")
        return rebuilt


class LiveGeometryHandoff:
    """Pass the original completed guard to its synchronous consuming observer."""

    def __init__(self):
        self._captured = None

    def capture(self, row):
        _require(self._captured is None, "prior geometry must be consumed before the next SDK step")
        self._captured = row, row["closure_geometry"]

    def take(self, row):
        info = self._captured
        _require(info is not None and info[0] is row and info[1] is row["closure_geometry"],
                 "same owned live geometry already evaluated by the original guard required")
        self._captured = None
        return info[1]


class PolicyOwnedPinchApproach(MeasuredPinchApproach):
    VERSION=VERSION
    def __init__(self,project,*,scope="owner_source_unqualified"):
        self.project=Path(project).resolve();self.scope=scope
        self.geometry=PolicyOwnedPinchGeometry(project,scope=scope)
        self.source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def capture(self,event,*,history_owner,grasp="pinch",live_geometry=None):
        try:return self._capture(event,history_owner=history_owner,grasp=grasp,live_geometry=live_geometry)
        except SourceFailure:raise
        except (ValueError,KeyError,TypeError,IndexError) as exc:
            raise SourceFailure("policy-owned Pinch source: "+str(exc)) from exc

    def _capture(self,event,*,history_owner,grasp,live_geometry=None):
        _require(grasp=="pinch","independent style-specific Pinch source required")
        from .tabletop_action24_history_owner import Action24HistoryOwner
        from .tabletop_native_teacher_action_history import NativeAction24HistoryOwner
        _require(type(history_owner) in (Action24HistoryOwner,NativeAction24HistoryOwner),"actual policy history owner required")
        history_owner._check();row=event["exchange"]
        compact = getattr(history_owner, "compact_batch", False)
        if compact:
            physical_identity = history_owner.claim_compact_physical(event)
        else:
            _require(history_owner._decision is not None and history_owner._row_hashes
                and history_owner._row_hashes[-1]==_digest(event),"actual durable row/history hash differs")
            _require(history_owner._target_hashes[-1]==_digest(
                row["action_readback"]["submission"]["expected_backend_targets"]),
                "actual submitted SDK target history differs")
            physical_identity = dict(physical_row_sha256=history_owner._row_hashes[-1],
                                     sdk_targets_sha256=history_owner._target_hashes[-1])
        decision=history_owner._decision
        _require(np.array_equal(np.asarray(row["action_readback"]["request"]["action24"]),
                                history_owner._issued.astype(np.float64)),
                 "actual issued policy request history differs")
        endpoint=row["action_readback"]["post"]["identity"][0]
        _require(all(history_owner._sequence.clock[k]==endpoint[k] for k in history_owner._sequence.clock),"same history endpoint/window clock required")
        if compact:
            _require(type(live_geometry) is LiveGeometryHandoff, "original live geometry handoff required")
            packet = live_geometry.take(row)
        else:
            _require(live_geometry is None, "archived evidence requires independent geometry verification")
            packet = self.geometry.validate(row["closure_geometry"], row)
        _require(packet["source_complete"],"required actual Pinch SDF distance source unavailable: "+str(packet["source_errors"]))
        result=self._metric(packet,_radius(row))
        binding=dict(collection_token=decision["policy"]["collection_token"],policy=deepcopy(decision["policy"]),
            history_owner_uid=history_owner._owner_uid,physical_samples=history_owner._sequence.substep,
            **physical_identity)
        result.update(version=VERSION,source_scope=self.scope,history_binding=binding)
        result["source"].update(version=VERSION,geometry_contract_version=GEOMETRY_VERSION,history_binding=binding,
            target_schema="fixed_material_nodes41_set_distance_v1",target_material_ids=list(range(41)))
        result["source_sha256"]=hashlib.sha256(_bytes(result["source"])).hexdigest()
        result["auxiliary"]["samples"]["approach_distance_m"]["source"]=(
            "policy_owned_SDK_pose_fixed41_locked_SDF:"+result["source_sha256"])
        return result
