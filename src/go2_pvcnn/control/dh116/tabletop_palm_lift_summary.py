"""Summarize measured palm displacement from each environment's reset height.

Inputs use metres and left/right column order. They are sensor telemetry only;
this helper neither derives a pose nor substitutes a value for unknown rows.
"""
import torch


@torch.no_grad()
def summarize_palm_lift(peak, last, valid, *, start=0, stop=None):
    """Return independent hand statistics over the half-open environment range.

    ``peak`` and ``last`` are floating tensors [N, 2], and ``valid`` is boolean
    with the same shape/device. Negative measurements remain negative. An empty
    valid set has None for its maximum, current mean and maximum environment ID.
    IDs are global; ties select the first valid environment. Only selected,
    valid measurements must be finite. No inputs or random state are modified.
    """
    if not all(isinstance(value, torch.Tensor) for value in (peak, last, valid)):
        raise ValueError("palm lift requires tensor inputs with shape [N, 2]")
    if (peak.ndim != 2 or peak.shape[1] != 2
            or last.shape != peak.shape or valid.shape != peak.shape):
        raise ValueError("palm lift tensors must share shape [N, 2]")
    if not peak.is_floating_point() or not last.is_floating_point():
        raise ValueError("palm lift measurements must be floating tensors")
    if valid.dtype != torch.bool:
        raise ValueError("palm lift validity must be a boolean tensor")
    if last.device != peak.device or valid.device != peak.device:
        raise ValueError("palm lift tensors must share a device")
    n = peak.shape[0]
    stop = n if stop is None else stop
    if type(start) is not int or type(stop) is not int or not 0 <= start <= stop <= n:
        raise ValueError("palm lift range must satisfy integer 0 <= start <= stop <= N")
    group_peak, group_last, group_valid = peak[start:stop], last[start:stop], valid[start:stop]
    if (not bool(torch.isfinite(group_peak[group_valid]).all())
            or not bool(torch.isfinite(group_last[group_valid]).all())):
        raise ValueError("valid palm lift measurements must be finite")
    result = {}
    for column, hand in enumerate(("left", "right")):
        mask = group_valid[:, column]
        local_ids = torch.nonzero(mask, as_tuple=False).flatten()
        count = local_ids.numel()
        maximum = mean = env_id = None
        if count:
            value, index = group_peak[:, column][mask].max(dim=0)
            maximum = float(value.item())
            mean = float(group_last[:, column][mask].mean().item())
            env_id = int(local_ids[index].item()) + start
        result.update({
            f"{hand}_palm_lift_max_m": maximum,
            f"{hand}_palm_lift_last_mean_m": mean,
            f"{hand}_palm_lift_max_env_id": env_id,
            f"{hand}_palm_lift_valid_envs": count,
        })
    return result
