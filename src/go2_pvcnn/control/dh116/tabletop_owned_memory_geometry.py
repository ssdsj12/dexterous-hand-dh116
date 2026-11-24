"""Internal owned snapshots; memory checks are not durable audit evidence."""
from copy import deepcopy
import math
from pathlib import PurePosixPath
import numpy as np


def owned_equal(a,b):
    """Compare every owned value without JSON, retaining array dtype and shape."""
    if isinstance(a,np.generic):a=a.item()
    if isinstance(b,np.generic):b=b.item()
    if hasattr(a,"detach"):a=a.detach().cpu().numpy()
    if hasattr(b,"detach"):b=b.detach().cpu().numpy()
    if isinstance(a,np.ndarray):
        return (isinstance(b,np.ndarray) and a.shape==b.shape and a.dtype==b.dtype
                and np.array_equal(a,b))
    if isinstance(b,np.ndarray):return False
    if isinstance(a,dict):
        return isinstance(b,dict) and a.keys()==b.keys() and all(owned_equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):
        return isinstance(b,(list,tuple)) and len(a)==len(b) and all(owned_equal(x,y) for x,y in zip(a,b))
    return type(a) is type(b) and a==b


class FrozenDict(dict):
    def _blocked(self,*args,**kwargs):
        raise TypeError("owned training snapshot is read-only")
    __setitem__=__delitem__=clear=pop=popitem=setdefault=update=__ior__=_blocked

    def __deepcopy__(self,memo):
        return self


def freeze_owned(value):
    """Freeze an exclusively owned SDK copy, not an arbitrary borrowed buffer."""
    if isinstance(value,np.ndarray):
        value.setflags(write=False)
        return value
    if isinstance(value,dict):
        return FrozenDict({k:freeze_owned(v) for k,v in value.items()})
    if isinstance(value,(list,tuple)):
        return tuple(freeze_owned(v) for v in value)
    if value is None or type(value) in (bool,int,float,str) or isinstance(value,np.generic):
        return value
    raise TypeError("unsupported owned geometry value: "+type(value).__name__)


def _post_snapshot(value):
    """Copy metadata containers, sharing arrays already owned and read-only."""
    if isinstance(value,np.ndarray):
        return value if not value.flags.writeable else np.array(value,copy=True)
    if isinstance(value,dict):return {k:_post_snapshot(v) for k,v in value.items()}
    if isinstance(value,list):return [_post_snapshot(v) for v in value]
    if isinstance(value,tuple):return tuple(_post_snapshot(v) for v in value)
    return value


class OwnedGeometryMode:
    """Opt-in training-only representation; all physical calculations inherited."""
    def capture_from_post(self,exchange,*,sim,bindings,read_clock,post,all_link_poses):
        """Consume the same synchronous finish post and already-read full poses.

        This opt-in path trusts the caller's single-writer finish transaction for
        q/qd/targets. It verifies its exact hand/view identity and clock, and never
        substitutes a prior frame's post. The audit capture path stays inherited.
        """
        from .tabletop_training_reward import SourceFailure
        from .tabletop_measured_closure_geometry import _identity,_array,CLOCK_FIELDS
        from .tabletop_deformable_bimanual_step import _bindings,SIDES
        from .tabletop_joint_target_readback import WRIST_LINEAR,WRIST_ANGULAR
        hand=view=binding=None
        try:
            expected=_identity(exchange);env=expected["env_id"]
            _bindings(bindings,(env,))
            if post is not exchange["action_readback"]["post"] or post["identity"]!=[expected]:
                raise ValueError("same completed finish post object and identity required")
            snapshot=_post_snapshot(post)
            poses=_array(all_link_poses,(2,23,7))
            def current():
                c=deepcopy(read_clock())
                if any(c.get(k)!=expected[k] for k in CLOCK_FIELDS):
                    raise ValueError("SDK/rod clock changed during reused-post geometry capture")
                if (sim.current_time_step_index!=expected["physics_step"]
                        or not math.isclose(float(sim.current_time),expected["timestamp_s"],abs_tol=2e-7,rel_tol=0)):
                    raise ValueError("actual SDK boundary differs from completed finish post")
            current()
            raw=dict(identity=expected,joint_readback=snapshot,hands={},
                rod_nodes_m=_array(exchange["rod"]["node_positions"],(1,41,3))[0])
            for hi,binding in enumerate(bindings):
                hand=binding.hand;view=hand.root_physx_view;side=binding.side
                joint=snapshot["hands"][side];joint_names=list(hand.joint_names)
                names=list(hand.body_names);paths=deepcopy(view.link_paths);prims=list(view.prim_paths)
                if (joint["instance_object_id"]!=id(hand) or joint["view_object_id"]!=id(view)
                        or joint["joint_names"]!=joint_names or joint["prim_paths"]!=prims
                        or joint["row_indices"]!=[0] or joint["mounted"] is not True
                        or binding.mount is None or binding.mount.ids!=[joint_names.index(n) for n in WRIST_LINEAR]
                        or binding.mount.angular_ids!=[joint_names.index(n) for n in WRIST_ANGULAR]):
                    raise ValueError("finish post actual joint/instance/mount mapping differs")
                pads={n:exchange["rod"]["mesh_bindings"][side+"."+n] for n in self._grids[side]}
                containers={str(PurePosixPath(p["prim_path"]).parent) for p in pads.values()}
                if len(containers)!=1:raise ValueError("one independent authored hand container required")
                container=next(iter(containers))
                if (len(names)!=23 or len(set(names))!=23 or paths!=[[container+"/"+n for n in names]]
                        or prims!=[binding.env_paths[env]] or not set(self._geometry[side]["source"])<=set(names)):
                    raise ValueError("actual 23-link identity or original source mapping differs")
                for name,pad in pads.items():
                    locked=self.sources[side]["sdf_binding"][name]
                    if (pad["side"]!=side or pad["body_name"]!=name or pad["body_index"]!=names.index(name)
                            or pad["prim_path"]!=paths[0][names.index(name)] or pad["sdf_sha256"]!=locked["sha256"]
                            or pad["capture_sha256"]!=locked["capture_sha256"]):
                        raise ValueError("same actual pad/SDF binding required")
                velocities=_array(view.get_link_velocities(),(1,23,6))
                if (view is not hand.root_physx_view or names!=list(hand.body_names) or paths!=view.link_paths
                        or prims!=list(view.prim_paths) or joint_names!=list(hand.joint_names)):
                    raise ValueError("actual link/joint identity changed during reused-post capture")
                raw["hands"][side]=dict(body_names=names,link_paths=paths,prim_paths=prims,
                    link_poses_xyzw=poses[hi:hi+1],link_velocities_world=velocities,
                    link_velocity_semantics="PhysX get_link_velocities world COM linear and angular velocity")
            current()
            if post is not exchange["action_readback"]["post"] or not owned_equal(snapshot,post):
                raise ValueError("completed finish post changed during geometry capture")
            return self._evaluate(raw,exchange)
        except SourceFailure:raise
        except (ValueError,KeyError,TypeError,IndexError) as exc:
            raise SourceFailure("policy-owned reused-post SDK source capture: "+str(exc)) from exc
        finally:
            hand=view=binding=None

    def _evaluate(self,raw,exchange):
        cache=getattr(self,"geometry_cache",None);sdf_cache=getattr(self,"sdf_cache",None)
        if cache is not None or sdf_cache is not None:
            poses=np.stack([raw["hands"][side]["link_poses_xyzw"][0] for side in ("left","right")])
            if tuple(tuple(raw["hands"][side]["body_names"]) for side in ("left","right"))!=self.geometry_body_names:
                raise ValueError("batch geometry actual body mapping changed")
            frame=self.radius_source.source._boundary()
            for active in (cache,sdf_cache):
                if active is not None:active.validate(self.geometry_lane,poses,raw["rod_nodes_m"],frame=frame)
        result=super()._evaluate(raw,exchange)
        if cache is not None:
            result["batched_geometry_used"]=True
            result["batched_geometry_precision"]="float64; near-boundary original CPU fallback"
        if sdf_cache is not None:
            result["batched_sdf_used"]=True
            result["batched_sdf_precision"]="original float32 local coordinates and interpolation"
        return result

    def _expected_radius(self):
        if not self.fast_memory:return super()._expected_radius()
        if not hasattr(self,"_fixed_expected_radius"):
            self._fixed_expected_radius=super()._expected_radius()
        return self._fixed_expected_radius

    def _pad_samples(self,side,matrices,nodes,*,radius):
        cache=getattr(self,"sdf_cache",None)
        if cache is None:return super()._pad_samples(side,matrices,nodes,radius=radius)
        lane=self.geometry_lane;s=0 if side=="left" else 1
        if cache.node_gaps.dtype!=np.dtype(np.float32) or float(cache.radii[lane])!=radius:
            raise ValueError("original float32 pad cache and same actual radius required")
        for name in self.REQUIRED_PADS:
            if (cache.links[s,name]!=self.geometry_body_names[s].index(name)
                    or not owned_equal(cache.source_bindings.get((s,name)),self.sources[side]["sdf_binding"][name])):
                raise ValueError("batched pad cache original link/SDF source binding differs")
        return {name:cache.row(lane,s,name) for name in self.REQUIRED_PADS}

    def _shape_metrics(self,vertices,matrix,*,side,key,nodes):
        cache=getattr(self,"geometry_cache",None)
        if cache is None:
            return super()._shape_metrics(vertices,matrix,side=side,key=key,nodes=nodes)
        i=self.geometry_lane;j=cache.columns[(0 if side=="left" else 1,key)]
        table=float(cache.minimum[i,j,2]-.3);lower=float(cache.lower[i,j])
        # This margin selects the original calculation; it never changes a guard threshold.
        from .tabletop_deformable_wrist_projection import CLEARANCE_M
        if min(abs(table-CLEARANCE_M),abs(lower),abs(float(cache.separation[i])))<1e-7:
            return super()._shape_metrics(vertices,matrix,side=side,key=key,nodes=nodes)
        return np.stack((cache.minimum[i,j],cache.maximum[i,j])),table,lower

    def _same_owned(self,a,b):
        return owned_equal(a,b) if self.fast_memory else super()._same_owned(a,b)

    def _seal_result(self,raw):
        if not self.fast_memory:return super()._seal_result(raw)
        if self._frozen_sources is None:self._frozen_sources=freeze_owned(self.sources)
        return dict(raw=freeze_owned(raw),raw_sha256=None,sources=self._frozen_sources,
                    evidence_mode="owned_memory_unhashed",durable_geometry_evidence=False)

    def validate(self,packet,exchange):
        if packet.get("evidence_mode")=="owned_memory_unhashed":
            from .tabletop_training_reward import SourceFailure
            raise SourceFailure("in-memory training geometry is not durable replay evidence")
        return super().validate(packet,exchange)
