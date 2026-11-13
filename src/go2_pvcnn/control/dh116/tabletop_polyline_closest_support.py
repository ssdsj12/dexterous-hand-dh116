"""Independent bounded closest-point direction proposals for support proofs.

Four optimization variables are a point in the ORIGINAL convex hull and an
adjacent rod segment fraction. Optimizer convergence is never a clearance
proof: complete vertex/segment support is independently recomputed.
No original geometry, material, radius, guard or default/native path changes.
"""
import math
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull,QhullError
from . import tabletop_polyline_support_certificate as base

VERSION="dh116-polyline-closest-support-cpu-v1"
SCALE=.01

def support(v,a,b,normal):
    n=np.asarray(normal,float)
    if n.shape!=(3,) or not np.isfinite(n).all() or np.linalg.norm(n)>1.:
        raise ValueError("finite nonexpansive direction required")
    return max(0.,float((v@n).min()-max(a@n,b@n)))-base.RADIUS-base.NUMERIC_GUARD

def proposal(v,a,b):
    origin=v.mean(0);vv=(v-origin)/SCALE
    hull=ConvexHull(vv);H=hull.equations[:,:3];h=hull.equations[:,3]
    aa=(a-origin)/SCALE;d=(b-a)/SCALE
    if np.any(h>=0):raise ValueError("full dimensional strict convex interior required")
    def diff(x):return x[:3]-aa-x[3]*d
    def fun(x):return float(diff(x)@diff(x))
    def jac(x):
        r=diff(x);return np.r_[2*r,-2*r@d]
    C=np.column_stack([H,np.zeros(len(H))])
    fit=minimize(fun,np.r_[np.zeros(3),.5],jac=jac,
        bounds=[*zip(vv.min(0),vv.max(0)),(0,1)],
        constraints=[dict(type="ineq",fun=lambda x:-C@x-h,jac=lambda x:-C)],
        method="SLSQP",options=dict(maxiter=64,ftol=1e-13))
    info=dict(status=int(fit.status),success=bool(fit.success),iterations=int(fit.nit),
        variables=4,hull_faces=len(H),convex_vertices=len(hull.vertices))
    if not fit.success or not np.isfinite(fit.x).all() or not 0<=fit.x[3]<=1:
        return dict(info,new_normal=None,reason="optimizer_did_not_return_a_finite_bounded_proposal")
    # Move the proposed point toward a verified strict interior, if necessary.
    # This only constructs a feasible upper-bound witness, never shrinks a hull.
    point=fit.x[:3].copy();hp=H@point;positive=hp>0
    factor=min(1.,float(np.min(-h[positive]/hp[positive]))) if positive.any() else 1.
    point*=max(0.,factor)*(1-1e-12)
    world=origin+SCALE*point;segment=a+fit.x[3]*(b-a)
    residual=world-segment;distance=float(np.linalg.norm(residual))
    normal=None if distance<=1e-14 else (residual/(distance*(1+1e-12))).tolist()
    return dict(info,new_normal=normal,point_world_m=world.tolist(),
        segment_fraction=float(fit.x[3]),segment_point_world_m=segment.tolist(),
        feasible_distance_upper_m=distance+base.NUMERIC_GUARD,
        numerical_upper_guard_m=base.NUMERIC_GUARD,
        reason="full_vertex_support_is_the_lower_bound_not_the_optimizer")

def refine(vertices,nodes,*,max_segments=4):
    v,p=base.arrays(vertices,nodes)
    if type(max_segments) is not int or not 1<=max_segments<=40:
        raise ValueError("explicit 1..40 CPU proposal budget required")
    old=base.certify(v,p);base.validate(old,v,p)
    candidates=set(sorted(range(len(p)-1),key=lambda i:old["segments"][i]["lower_m"])[:max_segments])
    rows=[];calls=0
    for i,prior in enumerate(old["segments"]):
        row=dict(segment=i,old_lower_m=prior["lower_m"],new_normal=None,proposal=None,
                 new_lower_m=None,lower_m=prior["lower_m"])
        if i in candidates:
            calls+=1
            try:pr=proposal(v,p[i],p[i+1])
            except QhullError:pr=dict(new_normal=None,reason="degenerate_hull_keeps_original_bound")
            row["proposal"]=pr;row["new_normal"]=pr["new_normal"]
            if row["new_normal"] is not None:
                row["new_lower_m"]=support(v,p[i],p[i+1],row["new_normal"])
                row["lower_m"]=max(row["old_lower_m"],row["new_lower_m"])
        rows.append(row)
    lower=max(old["old_global_xz_lower_m"],min(r["lower_m"] for r in rows))
    result=dict(version=VERSION,vertices_sha256=base.sha(v),nodes_sha256=base.sha(p),
        rod_radius_m=base.RADIUS,numeric_guard_m=base.NUMERIC_GUARD,
        original_certificate=old,segments=rows,lower_m=lower,clear=lower>=0,
        optimization_variables=4,optimizer_calls=calls,max_segments=max_segments,
        geometry_scope="original_source_convex_hull_and_all_original_adjacent_node_capsules",
        actual_contact_measurement=False,native_qualification=False)
    validate(result,v,p)
    return result

def validate(packet,vertices,nodes):
    v,p=base.arrays(vertices,nodes)
    if (packet.get("version")!=VERSION or packet.get("vertices_sha256")!=base.sha(v)
            or packet.get("nodes_sha256")!=base.sha(p) or packet.get("rod_radius_m")!=base.RADIUS
            or packet.get("numeric_guard_m")!=base.NUMERIC_GUARD
            or packet.get("optimization_variables")!=4
            or packet.get("actual_contact_measurement") is not False
            or packet.get("native_qualification") is not False
            or len(packet.get("segments",[]))!=len(p)-1):
        raise ValueError("exact full geometry and proof contract required")
    old=packet["original_certificate"];base.validate(old,v,p)
    bounds=[];calls=0
    for i,row in enumerate(packet["segments"]):
        prior=old["segments"][i]["lower_m"]
        if row["segment"]!=i or row["old_lower_m"]!=prior:raise ValueError("original segment order/bound differs")
        n=row["new_normal"];pr=row["proposal"]
        if pr is not None:
            calls+=1
            if pr["new_normal"]!=n:raise ValueError("direction proposal differs")
            if "point_world_m" in pr:
                point=np.asarray(pr["point_world_m"],float);alpha=pr["segment_fraction"]
                if point.shape!=(3,) or not np.isfinite(point).all() or not 0<=alpha<=1:
                    raise ValueError("finite feasible witness required")
                origin=v.mean(0);hull=ConvexHull((v-origin)/SCALE)
                if np.max(hull.equations[:,:3]@((point-origin)/SCALE)+hull.equations[:,3])>1e-9:
                    raise ValueError("claimed convex witness lies outside full source hull")
                segment=p[i]+alpha*(p[i+1]-p[i])
                if not np.allclose(segment,pr["segment_point_world_m"],atol=1e-12,rtol=0):
                    raise ValueError("segment witness differs")
                distance=float(np.linalg.norm(point-segment))
                if (pr["numerical_upper_guard_m"]!=base.NUMERIC_GUARD or
                    not math.isclose(pr["feasible_distance_upper_m"],distance+base.NUMERIC_GUARD,abs_tol=1e-12,rel_tol=0)):
                    raise ValueError("feasible distance witness differs")
        if n is None:
            if row["new_lower_m"] is not None:raise ValueError("no direction cannot provide a bound")
            expected=prior
        else:
            new=support(v,p[i],p[i+1],n)
            if not math.isclose(new,row["new_lower_m"],abs_tol=1e-12,rel_tol=0):
                raise ValueError("full vertex/segment support differs")
            expected=max(prior,new)
        if not math.isclose(expected,row["lower_m"],abs_tol=1e-12,rel_tol=0):
            raise ValueError("segment lower bound differs")
        bounds.append(expected)
    low=max(old["old_global_xz_lower_m"],min(bounds))
    if (not math.isclose(packet["lower_m"],low,abs_tol=1e-12,rel_tol=0)
            or packet["clear"]!=(low>=0) or calls!=packet["optimizer_calls"]
            or type(packet["max_segments"]) is not int or not 1<=packet["max_segments"]<=40
            or calls>packet["max_segments"]):
        raise ValueError("complete aggregate bound or budget differs")
    return True
