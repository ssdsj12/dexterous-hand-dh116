"""Adaptive joint-box coverage of a planned polyline, with a static wire only."""
import numpy as np


def certify_joint_path(waypoints, check_box, *, max_depth=18, progress=None):
    if type(max_depth) is not int or not 0<=max_depth<=20 or len(waypoints)<2:
        raise ValueError('invalid subdivision or waypoints')
    points=[]
    for item in waypoints:
        if set(item)!={'left','right'}:raise ValueError('both arms required')
        q={s:np.asarray(item[s],dtype=float) for s in item}
        if any(v.shape!=(13,) or not np.isfinite(v).all() for v in q.values()):
            raise ValueError('finite named 13D states required')
        points.append(q)
    covered=[];checks=0
    for leg,(start,end) in enumerate(zip(points[:-1],points[1:])):
        stack=[(0.,1.,0)]
        while stack:
            a,b,depth=stack.pop();u=(a+b)/2
            mid={s:(1-u)*start[s]+u*end[s] for s in start}
            delta={s:np.abs(end[s]-start[s])*(b-a)/2 for s in start}
            result=check_box(mid,delta,leg);checks+=1
            if result.get('joint_box_covered') is not True:
                raise ValueError('checker did not cover the entire joint box')
            if progress is not None and checks%20==0:
                progress(dict(checks=checks,leg=leg,covered_intervals=len(covered),current_start=a,current_end=b))
            passed=result.get('source_clearance_passed')
            if not isinstance(passed,(bool,np.bool_)):raise ValueError('invalid checker pass flag')
            if passed:
                bound=float(result['minimum_lower_bound_m'])
                if not np.isfinite(bound) or bound<.001:raise ValueError('invalid source certificate')
                covered.append(dict(leg=leg,start=a,end=b,minimum_lower_bound_m=bound))
            elif depth==max_depth:
                return dict(static_wire_source_sweep_passed=False,motion_qualified=False,checks=checks,
                    covered_intervals=covered,failed_interval=dict(leg=leg,start=a,end=b,result=result))
            else:
                stack.extend(((u,b,depth+1),(a,u,depth+1)))
    return dict(static_wire_source_sweep_passed=True,motion_qualified=False,checks=checks,
        covered_intervals=covered,failed_interval=None)
