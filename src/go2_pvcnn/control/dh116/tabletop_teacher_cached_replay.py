"""Replay actual per-link native audit poses without inferred SDK index mappings."""
import numpy as np

def audited_link_poses(row,bindings=None):
    records=row["display_audit"]["records"]
    grouped={}
    seen=set()
    for rec in records:
        key=(rec["env_id"],rec["hand"],rec["link"])
        if key in seen:raise ValueError("duplicate recorded link identity")
        seen.add(key)
        pose=np.asarray(rec["sdk_position_m"]+rec["sdk_quaternion_wxyz"],dtype=float)
        if pose.shape!=(7,) or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:])-1)>1e-5:
            raise ValueError("invalid native link pose")
        grouped.setdefault(key[:2],{})[key[2]]=pose
    if bindings is None:
        bindings=[dict(env_id=env,hand=side,body_names=list(links))
            for (env,side),links in sorted(grouped.items())]
    expected={(b["env_id"],b["hand"],name) for b in bindings for name in b["body_names"]}
    if expected!=seen:raise ValueError("native link identity changed within replay")
    poses={(b["env_id"],b["hand"]):np.stack(
        [grouped[(b["env_id"],b["hand"])][name] for name in b["body_names"]]) for b in bindings}
    return bindings,poses
