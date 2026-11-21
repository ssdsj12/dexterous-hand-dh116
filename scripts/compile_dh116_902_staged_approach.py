"""CPU-only Tripod/right phase compiler; not an executable native recipe.

The original 902 target remains unchanged. Measured wrist settling and real
contact/reaction qualification are required before a runtime may use this plan.
"""
import numpy as np
from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
from go2_pvcnn.control.dh116.linear_coupling import expand

VERSION = "dh116-902-staged-approach-cpu-v1"


def compile_candidate(base, initial_command, *, extra_middle_open_rad, phase_windows):
    a=np.asarray(base,dtype=float);q=np.asarray(initial_command,dtype=float).copy()
    held=list(range(6))+list(range(12,15))+list(range(18,21))
    if (a.shape!=(66,24) or q.shape!=(6,) or not np.isfinite(a).all()
            or not np.isfinite(q).all() or np.max(abs(a))>1. or np.any(a[:,held]!=0.)
            or not isinstance(phase_windows,(tuple,list)) or len(phase_windows)!=3
            or any(type(v) is not int or not 1<=v<=120 for v in phase_windows)
            or sum(phase_windows)>240 or type(extra_middle_open_rad) not in (int,float)
            or not np.isfinite(extra_middle_open_rad) or not 0<=extra_middle_open_rad<=.06):
        raise ValueError("bounded right-only 66-window source and explicit three-phase budget required")
    if any(np.any(np.linalg.norm(a[:,k:k+3],axis=1)>1.) for k in (15,21)):
        raise ValueError("source wrist vector would be normalized by the original controller")
    expand(q)
    q=q.astype(np.float32).astype(float);expand(q)
    n_open,n_move,n_close=phase_windows
    start_close=n_open+n_move
    scale=8/480*ACTION_CONTRACT["joint_rate_rad_s"]
    totals=a.sum(0)
    if totals[9]>=0.:
        raise ValueError("source must open the right middle finger; never reverse identity")
    result=np.zeros((sum(phase_windows),24))
    result[:n_open,9]=(totals[9]-extra_middle_open_rad/scale)/n_open
    wrist=[15,16,17,21,22,23]
    result[n_open:start_close,wrist]=totals[wrist]/n_move
    result[start_close:,6:12]=totals[6:12]/n_close
    result[start_close:,9]=extra_middle_open_rad/scale/n_close
    if np.max(abs(result))>1. or not np.allclose(result.sum(0),totals,atol=1e-12,rtol=0.):
        raise ValueError("phase schedule exceeds original action contract; clipping forbidden")
    if any(np.any(np.linalg.norm(result[:,k:k+3],axis=1)>1.) for k in (15,21)):
        raise ValueError("compiled wrist vector would be normalized by the original controller")
    nominal=result.copy();desired=q.copy();waypoints=[]
    for i in range(len(result)):
        desired+=nominal[i,6:12]*scale
        expand(desired)
        moving=nominal[i,6:12]!=0.
        corrected=np.zeros(6)
        corrected[moving]=(desired[moving]-q[moving])/scale
        if (np.max(abs(corrected-nominal[i,6:12]))*scale>1e-6
                or np.max(abs(corrected))>1.):
            raise ValueError("float32 compilation exceeds numerical budget")
        result[i,6:12]=corrected
        for _ in range(8):
            q=(q+corrected*scale/8).astype(np.float32).astype(float)
            expand(q)
        if np.max(abs(q-desired))>1e-6:
            raise ValueError("original waypoint unresolved after float32 writes")
        waypoints.append(q.tolist())
    return dict(version=VERSION,scope="ideal_command_candidate_only",actions=result,
        command_waypoints_rad=waypoints,phase_windows=list(phase_windows),
        phase_ranges=dict(open_middle=[0,n_open],move_wrist=[n_open,start_close],
                          close_fixed_wrist=[start_close,len(result)]),
        extra_middle_open_rad=float(extra_middle_open_rad),
        requires_measured_wrist_settling=True,
        close_success_requires="opposed_contact_and_delivered_reaction",
        native_admissible=False,teacher_admission=False)
