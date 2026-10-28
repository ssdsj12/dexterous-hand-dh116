"""Geometry and measured-load reporting for a simulation-only fingertip fixture."""
import numpy as np


def paired_thumb_setup():
    """Current training drive settings; only fixture distance differs per pair."""
    return dict(effort_nm=.2, mimic_frequency=400., active_speed=.3,
        passive_speed=2.97, position_iterations=64, velocity_iterations=16,
        cases=[dict(side=side, digit='thumb', load=load, plate_outward_m=distance)
               for side in ('left','right')
               for load,distance in (('unloaded',.1),('loaded',0.))])


def bounded_position_target(requested, actual, previous, dt, error_cap, rate=.05):
    """Limit requested position lead, then retain the original command slew cap.

    Does not clamp measured state. During rapid externally driven motion the
    slew bound has priority, so this is not a hard guarantee on tracking error.
    """
    if not np.isfinite([requested,actual,previous,dt,error_cap,rate]).all() or min(dt,error_cap,rate)<=0:
        raise ValueError('finite positions and positive dt, cap and rate required')
    target=np.clip(requested,actual-error_cap,actual+error_cap)
    return float(previous+np.clip(target-previous,-rate*dt,rate*dt))


def fixture_center(points, sign, *, gap=.0005, thickness=.004):
    points=np.asarray(points,dtype=float)
    if (points.ndim!=2 or points.shape[1]!=3 or not len(points)
            or not np.isfinite(points).all() or sign not in (-1,1)
            or not np.isfinite([gap,thickness]).all() or gap<=0 or thickness<=0):
        raise ValueError('finite vertices, X direction, positive gap/thickness required')
    center=(points.max(axis=0)+points.min(axis=0))/2
    center[0]=sign*(np.max(sign*points[:,0])+gap+thickness/2)
    return center


def summarize_hold(times, normal_force, mimic_error, *, max_sample_gap_s=.02):
    arrays=[np.asarray(x,dtype=float) for x in (times,normal_force,mimic_error)]
    t,f,e=arrays
    if (any(x.ndim!=1 or not np.isfinite(x).all() for x in arrays)
            or not (t.shape==f.shape==e.shape) or np.any(np.diff(t)<=0)
            or np.any(f<0) or np.any(e<0) or not np.isfinite(max_sample_gap_s) or max_sample_gap_s<=0):
        raise ValueError('aligned finite ordered times/nonnegative force and error required')
    if not len(t):return dict(samples=0,sustained_14n=False,window_seconds=0.)
    duration=float(t[-1]-t[0])
    max_gap=float(np.max(np.diff(t))) if len(t)>1 else 0.
    mean=float(f.mean());std=float(f.std())
    return dict(samples=len(t),window_seconds=duration,normal_mean_n=mean,
        normal_min_n=float(f.min()),normal_max_n=float(f.max()),normal_std_n=std,
        max_mimic_error_rad=float(e.max()),max_sample_gap_s=max_gap,
        sustained_14n=bool(duration>=.99 and f.min()>=14. and f.max()<=30.
            and max_gap<=max_sample_gap_s and e.max()<=.02 and std<=.1*max(mean,1e-9)))
