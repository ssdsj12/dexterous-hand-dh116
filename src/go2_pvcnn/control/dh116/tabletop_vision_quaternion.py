"""Explicit quaternion representation preprocessing for the unchanged 3844D actor.

Sensor artifacts keep their original values. Opt-in actors choose the same
representative for q and -q by making the largest-magnitude component positive;
ties select the first component. No pose normalization or extra feature is used.
"""
import torch


RAW_WRIST_QUATERNIONS = 'raw'
HEMISPHERE_WRIST_QUATERNIONS = 'wrist_quaternion_hemisphere_v1'


def validate_wrist_quaternion_preprocessing(mode):
    if mode not in (RAW_WRIST_QUATERNIONS, HEMISPHERE_WRIST_QUATERNIONS):
        raise ValueError('unsupported wrist quaternion preprocessing mode')
    return mode


def canonicalize_wrist_quaternion_history(observation):
    """Return an owned [N,3844] history, changing only two quaternion signs/frame.

    The sensor-history v1 layout has four 961D frames, with xyz+wxyz wrist poses
    at [920:934]. Zero quaternions in unfilled history remain zero. The maximum
    component rule also covers valid 180-degree rotations whose scalar is zero.
    Dtype/device and all non-quaternion bytes are preserved; inputs are untouched.
    """
    if (not isinstance(observation, torch.Tensor) or not observation.is_floating_point()
            or observation.ndim != 2 or observation.shape[1] != 3844):
        raise ValueError('wrist quaternion preprocessing requires floating [N,3844] history')
    result = observation.clone()
    frames = result.reshape(-1, 4, 961)
    for start in (923, 930):
        quat = frames[..., start:start + 4]
        if not bool(torch.isfinite(quat).all()):
            raise ValueError('wrist quaternion history must be finite')
        dominant = quat.gather(-1, quat.abs().argmax(-1, keepdim=True))
        sign = torch.where(dominant < 0, -1., 1.)
        frames[..., start:start + 4] = quat * sign
    return result
