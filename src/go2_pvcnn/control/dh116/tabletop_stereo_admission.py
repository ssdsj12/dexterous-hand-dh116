"""Compare an externally verified proof with actual stereo processing semantics.

This comparison does not verify native evidence or grant qualification.
"""
from copy import deepcopy

from .tabletop_stereo_preprocessing import processing_contract


def require_qualified_stereo_processing(verified_stereo, camera_config):
    actual = processing_contract(camera_config)
    if (
        not isinstance(verified_stereo, dict)
        or verified_stereo.get("calibration_sha256") != actual["calibration_sha256"]
    ):
        raise ValueError("formal stereo proof calibration differs from selected camera")
    measured = deepcopy(verified_stereo.get("depth_estimator_config"))
    if not isinstance(measured, dict):
        raise ValueError("formal matcher differs from verified raw stereo processing")
    measured.pop("image_processing_batch_size", None)
    if measured != actual["depth_estimator"]:
        raise ValueError("formal matcher differs from verified raw stereo processing")
    return actual
