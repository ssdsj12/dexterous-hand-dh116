"""Simulation-only tabletop rod geometry and free settling qualification."""
import numpy as np

TABLE_CENTER=(.6,0.,.25)
TABLE_SIZE=(.8,1.2,.10)
WIRE_RADIUS=.008
WIRE_LENGTH=.5


def settled_metrics(position,velocity,active_constraints):
    p=np.asarray(position,dtype=float);v=np.asarray(velocity,dtype=float)
    if p.shape!=(3,41) or v.shape!=p.shape or not np.isfinite(p).all() or not np.isfinite(v).all():
        return dict(passed=False,reason='invalid rod state')
    top=TABLE_CENTER[2]+TABLE_SIZE[2]/2
    bottom=float(p[2].min()-WIRE_RADIUS)
    support_error=float(np.max(np.abs(p[2]-WIRE_RADIUS-top)))
    speed=float(np.linalg.norm(v,axis=0).max())
    margin=float(np.min(np.asarray(TABLE_SIZE)[:2,None]/2-
        np.abs(p[:2]-np.asarray(TABLE_CENTER)[:2,None])-WIRE_RADIUS))
    return dict(passed=bool(not active_constraints and support_error<=.001 and speed<=.03 and margin>=0),
        min_surface_z_m=bottom,support_error_m=support_error,max_speed_m_s=speed,
        tabletop_margin_m=margin,active_constraints=list(active_constraints))
