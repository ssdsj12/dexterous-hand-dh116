"""Native Python scalars must survive evidence persistence."""
import importlib
import json
import numpy as np

def test_summary_is_strict_json_even_with_numpy_scalar_comparisons():
    m=importlib.import_module("scripts.audit_dh116_tripod_sdk_path")
    assert hasattr(m,"summarize_replay"),"replay summary conversion required"
    summary=m.summarize_replay({"physics_steps":8,"policy_steps":1},
        {"rows":[{"projection_delta_m":[1e-5,0,0],"sdk_target_joint_error_rad":0.,
                  "sdk_target_position_error_m":0.}]},
        {"proved":True},{"within_budget":False})
    assert json.loads(json.dumps(summary,allow_nan=False))["projection_changed_steps"]==1
