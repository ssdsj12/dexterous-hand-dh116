"""Independent simulator-truth check of initial sensor geometry; never actor data."""
import numpy as np
from .tabletop_depth_camera_probe import _numpy, _rotation, cylinder_surface_distances


class VisionGeometryError(RuntimeError):
    def __init__(self,message,report):
        super().__init__(message)
        self.report=report


def audit_initial_points(packet, device_world, *, radius=.008, length=.5):
    points=_numpy(packet['points'])
    valid=_numpy(packet['point_valid'])
    hands=_numpy(device_world.roots)
    poses=np.concatenate((_numpy(device_world.wp),_numpy(device_world.wq)),axis=1)
    rows=[]
    for i in range(len(points)):
        local=points[i,0,valid[i]]
        world=local@_rotation(hands[i,0,3:]).T+hands[i,0,:3]
        dist=cylinder_surface_distances(world,poses[i],radius,length)
        if not len(dist) or not np.isfinite(dist).all():
            raise RuntimeError(f'initial sensor geometry has no finite samples in env {i}')
        row=dict(env_id=i,point_count=len(dist),median_m=float(np.median(dist)),
            p95_m=float(np.quantile(dist,.95)),max_m=float(dist.max()))
        rows.append(row)
    # A wrong slot, frame or depth convention is a sensor failure, not missing data.
    failed=[row['env_id'] for row in rows if row['p95_m']>.003]
    report=dict(passed=not failed,scope='initial_validation_only_GT_never_returned_to_actor',
        tolerance_p95_m=.003,environments=rows,failed_env_ids=failed)
    if failed:
        raise VisionGeometryError(
            f'initial rendered point geometry exceeds 3mm p95 tolerance: {sorted(rows,key=lambda row:row["p95_m"],reverse=True)[:5]}',
            report)
    return report
