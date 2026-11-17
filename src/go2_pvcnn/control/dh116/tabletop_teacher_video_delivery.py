"""Delivery labels derived from same-clock native support evidence.

Raw recorder net_lift is treated solely as baseline-relative height. A delivered
net Lift is present only after support clearance/load qualification. This is
neither a complete grasp success evaluator nor a source of synthesized motion.
"""
import math

def measured_annotation(frame,row,env_id):
    for k in ("sdk_step","sdk_time","policy_step"):
        if frame[k]!=row[k]:raise ValueError("video/native ledger identity mismatch: "+k)
    height=float(frame["net_lift"])
    if not math.isfinite(height):raise ValueError("invalid measured height")
    contact=row["support_loads"][env_id]
    clearance=row["clearance"][env_id]
    valid=(contact.get("contact_data_valid") is True and contact.get("step_id")==frame["sdk_step"]
        and clearance.get("eligible") is True and clearance.get("status")=="clear")
    net=clearance.get("net_lift_m") if valid else None
    if net is not None and (not math.isfinite(net) or net<0):
        raise ValueError("invalid support-cleared net lift")
    return dict(height_delta_m=height,net_lift_m=net,
        support_status=clearance.get("status","NA") if contact.get("contact_data_valid") else "NA",
        counts_as_grasp_success=False,sdk_step=frame["sdk_step"],
        sdk_time=frame["sdk_time"],policy_step=frame["policy_step"])
