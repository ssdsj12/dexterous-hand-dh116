"""Owned CPU sensor frames and compact on-policy history references.

No latent cache, critic field, or lossy sensor conversion is accepted. RGB
uses uint8 only when decoding it is bit-exact to the supplied float32 /255
values; arbitrary float32 RGB and every other sensor tensor retain raw bytes.
"""
from dataclasses import fields
import hashlib
import itertools

import torch

from .tabletop_stereo_observation import StereoHistoryBatch


INPUT_KEYS=tuple(field.name for field in fields(StereoHistoryBatch))
_RGB=('left_rgb','right_rgb')
_GENERATION_STRIDE=1<<32
_GENERATIONS=itertools.count(1)


def _identity(tensor):
    value=tensor.detach().contiguous()
    return (str(value.dtype),tuple(value.shape),hashlib.sha256(value.numpy()).digest())


class IndexedStereoRolloutStorage:
    """Storage hooks consumed by PPOTrainer, independent of its PPO equations.

    ``capture`` owns exact CPU sensor frames and returns int64[B] history IDs.
    ``resolve`` materializes only explicitly requested IDs on the target device.
    Explicit diagnostics may request a large CPU batch; PPOTrainer bounds all
    collection, frozen checks, and gradient forwards by ``max_visual_batch``.
    """
    def __init__(self, *, max_visual_batch=1):
        if type(max_visual_batch) is not int or max_visual_batch<1:
            raise ValueError('max_visual_batch must be a positive integer')
        self.max_visual_batch=max_visual_batch
        self._generation=0
        self.reset()

    def reset(self):
        self._generation=next(_GENERATIONS)
        self._images=[];self._image_ids={}
        self._frames=[];self._frame_ids={}
        self._samples=[]
        self._contract=None
        self._rgb_uint8_count=self._rgb_float32_count=0
        self._max_resolved_batch=0

    def capture(self, value):
        if isinstance(value,dict): value=StereoHistoryBatch.from_mapping(value)
        if type(value) is not StereoHistoryBatch:
            raise TypeError('indexed rollout accepts exact sensor history only; privileged mappings forbidden')
        value.validate()
        contract=tuple((key,tuple(tensor.shape[1:]),tensor.dtype) for key,tensor in value.as_dict().items())
        if self._contract is not None and contract!=self._contract:
            raise ValueError('sensor history shape/dtype changed within on-policy rollout')
        self._contract=contract
        # A CPU input is read in place. A device input transfers one frame at a
        # time, so staging cannot allocate a second complete rollout history.
        arrays={key:tensor.detach() for key,tensor in value.as_dict().items()}
        batch,timesteps=value.history_valid.shape
        refs=[]
        for row in range(batch):
            frame_ids=[]
            for tick in range(timesteps):
                non_rgb={key:tensor[row,tick].cpu() for key,tensor in arrays.items() if key not in _RGB}
                metadata=tuple(_identity(non_rgb[name]) for name in ('dt','image_age','image_valid','history_valid'))
                image_ids=[]
                for key in _RGB:
                    rgb=arrays[key][row,tick].cpu()
                    # Identity includes the view and all available image clock/
                    # validity metadata, independently of proprio and depth.
                    identity=(key,_identity(rgb),metadata)
                    image_id=self._image_ids.get(identity)
                    if image_id is None:
                        encoded=(rgb*255).round().to(torch.uint8)
                        decoded=encoded.to(torch.float32)/255
                        lossless=torch.equal(decoded.view(torch.uint8),rgb.contiguous().view(torch.uint8))
                        stored=encoded.clone() if lossless else rgb.clone()
                        image_id=len(self._images)
                        self._images.append(stored)
                        self._image_ids[identity]=image_id
                        if lossless:self._rgb_uint8_count+=1
                        else:self._rgb_float32_count+=1
                    image_ids.append(image_id)
                identity=(tuple(image_ids),tuple((key,_identity(tensor)) for key,tensor in non_rgb.items()))
                frame_id=self._frame_ids.get(identity)
                if frame_id is None:
                    frame_id=len(self._frames)
                    self._frames.append((tuple(image_ids),{key:tensor.clone() for key,tensor in non_rgb.items()}))
                    self._frame_ids[identity]=frame_id
                frame_ids.append(frame_id)
            sample_id=len(self._samples)
            self._samples.append(torch.tensor(frame_ids,dtype=torch.int64))
            refs.append(self._generation*_GENERATION_STRIDE+sample_id)
        return torch.tensor(refs,dtype=torch.int64)

    def resolve(self, references, device='cpu'):
        if (not isinstance(references,torch.Tensor) or references.dtype!=torch.int64
                or references.ndim!=1 or not references.numel()):
            raise ValueError('compact int64[B] history references required')
        ids=references.detach().cpu()
        generation=torch.div(ids,_GENERATION_STRIDE,rounding_mode='floor')
        indices=ids.remainder(_GENERATION_STRIDE)
        if bool((generation!=self._generation).any()) or bool((indices>=len(self._samples)).any()):
            raise ValueError('stale or foreign sensor history reference generation')
        self._max_resolved_batch=max(self._max_resolved_batch,len(ids))
        output={key:[] for key in INPUT_KEYS}
        for index in indices.tolist():
            frames=[self._frames[i] for i in self._samples[index].tolist()]
            for key in INPUT_KEYS:
                if key in _RGB:
                    view=_RGB.index(key)
                    values=[]
                    for image_ids,_ in frames:
                        rgb=self._images[image_ids[view]]
                        values.append(rgb.float()/255 if rgb.dtype==torch.uint8 else rgb)
                else: values=[record[key] for _,record in frames]
                output[key].append(torch.stack(values))
        # stack owns every result, even when all samples refer to the same frame.
        return {key:torch.stack(values).to(device) for key,values in output.items()}

    def stats(self):
        tensors=list(self._images)+self._samples
        tensors.extend(value for _,record in self._frames for value in record.values())
        return dict(image_count=len(self._images),frame_count=len(self._frames),sample_count=len(self._samples),
            rgb_uint8_count=self._rgb_uint8_count,rgb_float32_count=self._rgb_float32_count,
            storage_bytes=sum(t.numel()*t.element_size() for t in tensors),
            max_resolved_batch=self._max_resolved_batch,device='cpu',
            identity='SHA256 of exact dtype/shape/raw bytes and available sensor metadata',
            evidence_scope='software_storage_only_not_native_capture')
