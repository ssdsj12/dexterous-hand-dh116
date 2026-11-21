"""Read-only reserve bounds; neither a native guard replacement nor a tracking proof."""
import numpy as np

from scripts.audit_dh116_certified_segment_distance import segment_hull_bounds
from scripts.audit_dh116_pinch_initial_pose import box_relation
from scripts.screen_dh116_tripod_supported_reset import archived_tube

ROUNDING_M = 1e-12
CONTACT_FLOOR_M = -.00015
SUPPORT_FLOOR_M = .001


def _vertices(value):
    v=np.asarray(value,dtype=float)
    if (v.ndim!=2 or v.shape[1]!=3 or len(v)<4 or not np.isfinite(v).all()
            or np.linalg.matrix_rank(v-v.mean(0))<3):
        raise ValueError("finite full-dimensional convex vertices required")
    return v


def _reserve(value):
    if type(value) not in (float,int) or not np.isfinite(value) or not 0<=value<=.01:
        raise ValueError("finite bounded nonnegative requested reserve required")
    return float(value)


def _summary(lower,upper,floor,reserve):
    if not np.isfinite([lower,upper]).all() or lower>upper:
        raise ValueError("finite ordered independently certified bounds required")
    target=floor+reserve
    passed=lower>=target
    insufficient=upper<target
    return dict(lower_m=float(lower),upper_m=float(upper),guard_floor_m=floor,
        requested_reserve_m=reserve,requested_floor_m=target,
        guard_certified=bool(lower>=floor),guard_violated=bool(upper<floor),
        requested_reserve_certified=bool(passed),upper_proves_insufficient=bool(insufficient),
        reserve_status="certified" if passed else "insufficient" if insufficient else "unknown",
        native_qualification=False,teacher_admission=False)


def box_reserve_certificate(vertices,low,high,*,requested_reserve_m):
    """AABB/SAT are lower bounds; feasible vertex/box pairs are upper bounds."""
    v=_vertices(vertices);lo=np.asarray(low,dtype=float);hi=np.asarray(high,dtype=float)
    reserve=_reserve(requested_reserve_m)
    if lo.shape!=(3,) or hi.shape!=(3,) or not np.isfinite([lo,hi]).all() or not (lo<hi).all():
        raise ValueError("finite nonempty support box required")
    # Translate to avoid cancellation in the support-function calculations.
    origin=(lo+hi)/2;v=v-origin;lo=lo-origin;hi=hi-origin
    fast=max(0.,float(np.linalg.norm(np.maximum(np.maximum(lo-v.max(0),v.min(0)-hi),0.)))-ROUNDING_M)
    lower=fast;sat=None
    if fast<SUPPORT_FLOOR_M+reserve:
        sat=float(box_relation(v,(lo+hi)/2,hi-lo)["max_separating_axis_gap_m"])-ROUNDING_M
        lower=max(lower,sat)
    upper=float(np.linalg.norm(v-np.clip(v,lo,hi),axis=1).min())+ROUNDING_M
    return dict(_summary(lower,upper,SUPPORT_FLOOR_M,reserve),
        method="aabb_sat_lower_feasible_pair_upper",fast_lower_m=fast,sat_lower_m=sat)


def rod_reserve_certificate(vertices,node_samples,*,requested_reserve_m,max_qp_calls=40):
    """Certify a requested reserve from all nine-frame, 3-node convex envelopes.

    Last-frame finite-segment witnesses supply directions and valid upper bounds
    only. Every lower bound uses ALL frames and vertices of its original group.
    A bounded failure leaves an unresolved reserve; it never relaxes the guard.
    """
    v=_vertices(vertices);nodes=np.asarray(node_samples,dtype=float)
    reserve=_reserve(requested_reserve_m)
    if (nodes.shape!=(9,41,3) or not np.isfinite(nodes).all()
            or type(max_qp_calls) is not int or not 0<=max_qp_calls<=40):
        raise ValueError("complete finite nine-frame envelope and at most 40 QP calls required")
    origin=v.mean(0);v=v-origin;nodes=nodes-origin
    normals,support=archived_tube(nodes)
    lower=np.max(np.min(v@normals.T,axis=0)[None,:]-support,axis=1)-ROUNDING_M
    fast=float(lower.min())
    # These are feasible points in the full hull/envelope union, hence an upper bound.
    upper=float(np.linalg.norm(v[:,None,:]-nodes.reshape(-1,3)[None,:,:],axis=2).min())-.005+ROUNDING_M
    target=CONTACT_FLOOR_M+reserve;calls=0;witnesses=[];reason="completed"
    for group in np.argsort(lower,kind="stable"):
        if lower[group]>=target:continue
        start=int(group)*2;full_group=nodes[:,start:start+3].reshape(-1,3)
        for segment in (start,start+1):
            if calls>=max_qp_calls:
                reason="qp_budget_exhausted";break
            calls+=1
            try:
                bound=segment_hull_bounds(v,nodes[-1,segment],nodes[-1,segment+1])
            except ValueError as error:
                reason="distance_certificate_failed: "+str(error);break
            upper=min(upper,bound["upper_m"]-.005)
            delta=np.asarray(bound["hull_point_m"])-np.asarray(bound["segment_point_m"])
            norm=float(np.linalg.norm(delta))
            if not np.isfinite(norm) or norm==0.:
                reason="zero_separating_direction";break
            normal=delta/(norm*(1.+1e-12))
            value=float(np.min(v@normal)-np.max(full_group@normal)-.005-ROUNDING_M)
            lower[group]=max(lower[group],value)
            witnesses.append(dict(group=int(group),segment=segment,normal=normal.tolist(),
                source_frame=8,support_lower_m=value,segment_upper_m=bound["upper_m"]-.005))
            if lower[group]>=target:break
        if reason!="completed":break
    return dict(_summary(float(lower.min()),upper,CONTACT_FLOOR_M,reserve),
        method="full_nine_frame_support_lower_feasible_subset_upper",fast_lower_m=fast,
        group_lower_m=lower.tolist(),frame_count=9,radius_m=.005,qp_calls=calls,
        reason=reason,witnesses=witnesses)


def main(argv=None):
    import argparse
    import hashlib
    import json
    from pathlib import Path
    from scripts.diagnose_dh116_teacher_initialization import write_new
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--witnesses",type=Path,required=True)
    parser.add_argument("--sha256",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--rod-reserve-m",type=float,default=.0001)
    parser.add_argument("--support-reserve-m",type=float,default=.0015)
    args=parser.parse_args(argv)
    content=args.witnesses.read_bytes()
    if hashlib.sha256(content).hexdigest()!=args.sha256:
        raise ValueError("original witness SHA mismatch")
    packet=json.loads(content)
    rows=packet.get("witness_geometry",[])
    total=packet.get("SDK_substeps")
    if (packet.get("scope")!="staged_ideal_command_CPU_only_not_native_receipt"
            or type(total) is not int or total<=0 or not 1<=len(rows)<=16
            or len({row.get("ordinal") for row in rows})!=len(rows)
            or any(type(row.get("ordinal")) is not int or not 1<=row["ordinal"]<=total for row in rows)):
        raise ValueError("bounded distinct ideal-command witness ordinals required")
    result=[]
    contacts={group+":"+pad for group in ("source_colliders","cooked_pads")
        for pad in ("finger14_link","finger23_link","finger33_link")}
    for row in rows:
        shapes=row["shapes"]
        if len(shapes)!=22 or not contacts<=set(shapes) or len(row["boxes"])!=2:
            raise ValueError("all 22 original hand shapes and both supports required")
        rods={name:rod_reserve_certificate(shapes[name],row["rod_samples"],
            requested_reserve_m=args.rod_reserve_m) for name in sorted(contacts)}
        supports={name+":"+str(i):box_reserve_certificate(v,lo,hi,
            requested_reserve_m=args.support_reserve_m)
            for name,v in sorted(shapes.items()) for i,(lo,hi) in enumerate(row["boxes"])}
        result.append(dict(ordinal=row["ordinal"],rod=rods,support=supports))
    import sys
    source_paths=[Path(__file__),*(Path(sys.modules[fn.__module__].__file__)
        for fn in (segment_hull_bounds,box_relation,archived_tube))]
    output=dict(version="dh116-902-tracking-reserve-audit-v1",
        scope="selected_ideal_command_geometry_only",witness_sha256=args.sha256,
        sampled_substeps=len(rows),total_substeps=total,full_path_certified=False,
        robust_route_qualified=False,native_qualification=False,teacher_admission=False,
        native_runs=0,records=result,
        source_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths})
    write_new(args.output,output)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
