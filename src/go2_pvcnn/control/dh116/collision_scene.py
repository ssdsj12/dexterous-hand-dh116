"""Complete DH116/Panda source-convex static screen, NOT motion approval.

Target pads may be delegated to a separate front-contact gate explicitly.
No other wire contact is exempt. Runtime/cooked/swept qualification is separate.
"""
from itertools import combinations, product
import numpy as np
from scipy.spatial import ConvexHull
from pxr import Usd, UsdGeom, UsdPhysics
from .usd_kinematics import SideKinematics
from .usd_surface import read_surface, layer_pins
from .clearance import hull_wire_lower_bound, separation_lower_bound,axis_lower_bound
from .wire_table import TABLE_CENTER, TABLE_SIZE, WIRE_RADIUS
from .linear_coupling import coupling_matrix, active_limits
from .sweep_bounds import point_rotation_radii, relative_motion_bound
from .clearance_certificate import ClearanceCertificate


class SourceCollisionScene:
    def __init__(self, stage, *, cache_certificates=False):
        if type(cache_certificates) is not bool:raise ValueError('explicit certificate cache flag required')
        self.cache_certificates=cache_certificates
        self._clearance_certificate=None
        self.layer_pins=layer_pins(stage)
        self.models={s:SideKinematics(stage,s) for s in ('left','right')}
        self.geometry={};self.empty_fixed_flanges=[];self.pad_bodies=set();self.limits={}
        edges=[]
        for side,model in self.models.items():
            edges+=model.edges
            bounds={}
            for prim in stage.Traverse():
                if prim.IsA(UsdPhysics.RevoluteJoint) and prim.GetName() in model.names[:7]:
                    j=UsdPhysics.RevoluteJoint(prim)
                    bounds[prim.GetName()]=np.deg2rad([j.GetLowerLimitAttr().Get(),j.GetUpperLimitAttr().Get()])
            self.limits[side]=np.array([bounds[n] for n in model.names[:7]])
            if not np.isfinite(self.limits[side]).all():raise ValueError('missing arm limits')
            for body,path in model.bodies.items():
                if body==side+'_panda_link8':
                    if any(p.IsA(UsdGeom.Gprim) for p in Usd.PrimRange(stage.GetPrimAtPath(path),Usd.TraverseInstanceProxies())):
                        raise ValueError('expected empty fixed flange now has geometry')
                    if not any(e[1]==path and e[2]==model.bodies['base_link'] and e[5] is None for e in model.edges):
                        raise ValueError('flange not fixed to this DH116 base')
                    self.empty_fixed_flanges.append(path);continue
                m=read_surface(stage,path)
                if m['collision_approximation']!='convexHull':raise ValueError('unsupported collision approximation')
                p=np.unique(np.array(m['points_body']),axis=0);h=ConvexHull(p)
                self.geometry[path]=dict(vertices=p[h.vertices],planes=h.equations,side=side,
                    source_mesh_path=m['mesh_path'],geometry_sha256=m['geometry_sha256'],
                    motion_radii=point_rotation_radii(model.edges,path,p[h.vertices],model.names))
                if body in ('finger14_link','finger23_link','finger33_link'):self.pad_bodies.add(path)
        # Same rigid cluster and directly joint-connected clusters are adjacent.
        groups={p:{p} for model in self.models.values() for p in model.bodies.values()}
        for e in edges:
            if e[5] is not None:continue
            group=groups[e[1]]|groups[e[2]]
            for p in group:groups[p]=group
        self.adjacent=set()
        for group in groups.values():
            self.adjacent.update(frozenset(pair) for pair in combinations(group,2))
        for e in edges:
            self.adjacent.update(frozenset((a,b)) for a in groups[e[1]] for b in groups[e[2]])
        center=np.array(TABLE_CENTER);half=np.array(TABLE_SIZE)/2
        self.table=np.array(list(product(*zip(center-half,center+half))))
        self._separation_axes={}

    def _separation(self,key,a,b,axes,required):
        # A cached axis is merely a candidate separating direction. Project
        # EVERY current vertex again. A stale clearance/pose is never reused.
        if key in self._separation_axes:
            bound=axis_lower_bound(a,b,self._separation_axes[key])
            if bound>=required:return bound
        bound,axis=separation_lower_bound(a,b,axes,return_axis=True)
        self._separation_axes[key]=axis
        return bound

    def check(self, states, wire_nodes, *, permit_pad_contact=False, margin=.001,
              active_half_delta=None, measured_body_poses=None,physical_motion_delta=None,
              wire_motion_bound=0.,wire_radius=WIRE_RADIUS, runtime_epsilon=False):
        if set(states)!=set(self.models) or type(permit_pad_contact) is not bool:
            raise ValueError('both sides and explicit boolean pad-contact mode required')
        if type(runtime_epsilon) is not bool:
            raise ValueError('runtime epsilon flag must be explicit')
        if not np.isfinite(margin) or margin < (.0009 if runtime_epsilon else .001):
            raise ValueError('minimum source guard margin is 1 mm')
        if not np.isfinite([wire_motion_bound,wire_radius]).all() or wire_motion_bound<0 or wire_radius<=0:
            raise ValueError('invalid wire motion bound/radius')
        if physical_motion_delta is not None and measured_body_poses is None:
            raise ValueError('physical motion envelope requires measured body poses')
        wire=np.asarray(wire_nodes,dtype=float)
        if wire.ndim!=2 or wire.shape[1]!=3 or len(wire)<2 or not np.isfinite(wire).all():
            raise ValueError('invalid wire nodes')
        poses={};full_delta={}
        if active_half_delta is not None and set(active_half_delta)!=set(self.models):
            raise ValueError('both sweep sides required')
        for side,model in self.models.items():
            q=np.asarray(states[side],dtype=float);bounds=self.limits[side]
            if (q.shape!=(13,) or not np.isfinite(q).all()
                    or np.any(q[:7]<bounds[:,0]) or np.any(q[:7]>bounds[:,1])):
                raise ValueError('invalid/out-of-limit active state')
            poses.update(model.fk(q)[0])  # also validates DH116 coupled limits
            d=np.zeros(13) if active_half_delta is None else np.asarray(active_half_delta[side],dtype=float)
            if d.shape!=(13,) or not np.isfinite(d).all() or np.any(d<0):
                raise ValueError('invalid active half-delta')
            hand_low,hand_high=active_limits();low=np.r_[bounds[:,0],hand_low];high=np.r_[bounds[:,1],hand_high]
            if np.any(q-d<low) or np.any(q+d>high):raise ValueError('swept joint box exceeds limits')
            full_delta[side]=np.r_[d[:7],np.abs(coupling_matrix())@d[7:]]
        if measured_body_poses is not None:
            if active_half_delta is not None:
                raise ValueError('measured poses cannot certify a nominal joint box')
            if set(measured_body_poses)!=set(self.geometry):
                raise ValueError('measured poses must cover every collision body exactly')
            for path,value in measured_body_poses.items():
                t=np.asarray(value,dtype=float)
                if (t.shape!=(4,4) or not np.isfinite(t).all()
                        or not np.allclose(t[3],[0,0,0,1],atol=1e-8,rtol=0)
                        or not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-5,rtol=0)
                        or not np.isclose(np.linalg.det(t[:3,:3]),1,atol=1e-5,rtol=0)):
                    raise ValueError('invalid measured rigid body pose')
                poses[path]=t
            if physical_motion_delta is not None:
                if set(physical_motion_delta)!=set(self.models):raise ValueError('both physical motion sides required')
                for side,model in self.models.items():
                    d=np.asarray(physical_motion_delta[side],dtype=float)
                    if d.shape!=(len(model.names),) or not np.isfinite(d).all() or np.any(d<0):
                        raise ValueError('invalid physical motion envelope')
                    full_delta[side]=d
        world={p:(m['vertices']@poses[p][:3,:3].T+poses[p][:3,3],
                  m['planes'][:,:3]@poses[p][:3,:3].T) for p,m in self.geometry.items()}
        context=(permit_pad_contact,physical_motion_delta is not None,
                 tuple((s,full_delta[s].tobytes()) for s in sorted(full_delta)))
        if self.cache_certificates and measured_body_poses is not None and self._clearance_certificate is not None:
            certificate,old_context,old_result=self._clearance_certificate
            if context==old_context and wire.shape==certificate.wire.shape:
                bound=certificate.lower_bound({p:v for p,(v,n) in world.items()},wire,
                    wire_radius=wire_radius,wire_motion=wire_motion_bound)
                if bound>=margin:
                    return dict(old_result,minimum_lower_bound_m=bound,certificate_reused=True,
                                wire_motion_bound_m=wire_motion_bound,wire_radius_m=wire_radius)
        boxes={};failures=[];minimum=np.inf
        table_lo=self.table.min(0);table_hi=self.table.max(0);wire_lo=wire.min(0);wire_hi=wire.max(0)
        def record(kind,body,bound,other=None):
            nonlocal minimum
            minimum=min(minimum,bound)
            if bound<margin:failures.append(dict(kind=kind,body=body,other=other,lower_bound_m=float(bound)))
        for path,m in self.geometry.items():
            t=poses[path];v,n=world[path]
            lo=v.min(0);hi=v.max(0);boxes[path]=(lo,hi)
            motion=float(m['motion_radii']@full_delta[m['side']])
            table_bound=float(np.maximum(lo-table_hi,table_lo-hi).max())
            if table_bound-motion<margin:table_bound=self._separation(('table',path),v,self.table,np.r_[n,np.eye(3)],margin+motion)
            record('table',path,table_bound-motion)
            if permit_pad_contact and path in self.pad_bodies:continue
            local=(wire-t[:3,3])@t[:3,:3]
            wire_bound=float(np.maximum(lo-wire_hi,wire_lo-hi).max()-wire_radius)
            if wire_bound-motion-wire_motion_bound<margin:
                wire_bound=hull_wire_lower_bound(m['planes'],local,wire_radius,.0005,
                    refine_below=margin+motion+wire_motion_bound)
            record('wire',path,wire_bound-motion-wire_motion_bound)
        for (pa,(a,na)),(pb,(b,nb)) in combinations(world.items(),2):
            if frozenset((pa,pb)) in self.adjacent:continue
            ma=self.geometry[pa];mb=self.geometry[pb]
            if ma['side']==mb['side']:
                motion=relative_motion_bound(ma['motion_radii'],mb['motion_radii'],full_delta[ma['side']],same_tree=True)
            else:motion=float(ma['motion_radii']@full_delta[ma['side']]+mb['motion_radii']@full_delta[mb['side']])
            alo,ahi=boxes[pa];blo,bhi=boxes[pb]
            bound=float(np.maximum(alo-bhi,blo-ahi).max())
            if bound-motion<margin:bound=self._separation(('pair',pa,pb),a,b,np.r_[na,nb,np.eye(3)],margin+motion)
            record('robot_pair',pa,bound-motion,pb)
        result=dict(source_clearance_passed=not failures,minimum_lower_bound_m=float(minimum),
            failures=failures,collision_bodies=len(world),motion_qualified=False,
            measured_body_poses_used=measured_body_poses is not None,
            measured_interval_bound_used=physical_motion_delta is not None,
            wire_motion_bound_m=float(wire_motion_bound),wire_radius_m=float(wire_radius),
            joint_box_covered=active_half_delta is not None,
            pad_front_contact_qualified=False,
            delegated_pad_bodies=sorted(self.pad_bodies) if permit_pad_contact else [],certificate_reused=False)
        if self.cache_certificates:
            self._clearance_certificate=None
            if not failures and measured_body_poses is not None:
                cert=ClearanceCertificate({p:v for p,(v,n) in world.items()},wire,
                    lower_bound=minimum,wire_radius=wire_radius,wire_motion=wire_motion_bound)
                self._clearance_certificate=(cert,context,dict(result))
        return result
