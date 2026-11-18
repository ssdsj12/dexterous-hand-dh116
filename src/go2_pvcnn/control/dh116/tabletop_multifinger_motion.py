"""Measured wrist-target cancellation; no physical state writes or resets."""
import torch

def cancel_queued_ascent(queued,measured_offsets,paused):
    if (queued.ndim!=3 or tuple(queued.shape[1:])!=(2,3) or measured_offsets.shape!=queued.shape
            or paused.shape!=(len(queued),) or paused.dtype!=torch.bool
            or measured_offsets.dtype!=queued.dtype or measured_offsets.device!=queued.device
            or paused.device!=queued.device or not queued.is_floating_point()
            or not torch.isfinite(queued).all() or not torch.isfinite(measured_offsets).all()):
        raise ValueError('finite matching N x 2 x 3 offsets and N pause mask required')
    result=queued.clone()
    result[:,:,2]=torch.where(paused[:,None],torch.minimum(queued[:,:,2],measured_offsets[:,:,2]),queued[:,:,2])
    return result
