from pathlib import Path
from types import SimpleNamespace
import ast
import pytest


def test_policy_configuration_accepts_measured_teacher_requests():
    from go2_pvcnn.control.dh116.tabletop_policy_control import validate_configuration
    args=SimpleNamespace(tensor_pipeline=True,palm_transport='driven_carriage',pressure_relief=True,
        skip_hold=True,grasp_readiness='measured_load',palm_exploration_axis='xyz',reset_height_jitter_m=0,
        teacher_search_cases=Path('candidates.json'),force_region_control=True,policy_lift_hold=True)
    validate_configuration(args)


def test_legacy_policy_teacher_without_force_contract_remains_rejected():
    from go2_pvcnn.control.dh116.tabletop_policy_control import validate_configuration
    args=SimpleNamespace(tensor_pipeline=True,palm_transport='driven_carriage',pressure_relief=True,
        skip_hold=True,grasp_readiness='measured_load',palm_exploration_axis='xyz',reset_height_jitter_m=0,
        teacher_search_cases=Path('candidates.json'),force_region_control=False,policy_lift_hold=False)
    with pytest.raises(ValueError):validate_configuration(args)
