"""Pinned monocular WireSeg supervision for the existing shared stereo backbone."""

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
from tempfile import NamedTemporaryFile

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from .tabletop_stereo_encoder import ResNet18, StereoEncoder


WIRESEG_REPOSITORY = "DeformX/WireSeg-36K"
WIRESEG_REVISION = "1dadc86f4e988a015cd9fd02b6dfd16c11c9c40b"
TRANSFER_CONTRACT = "wireseg-monocular-shared-stereo-backbone-v1"
# Hold out entire families while the released rows lack validated render-group IDs.
SPLITS = {
    "flying_wire": "train",
    "wire_on_plane": "validation",
    "data_center": "test",
    "real_easy": "real_test",
    "real_hard": "real_test",
}


def sample_split(family):
    if family not in SPLITS:
        raise ValueError(f"unrecognized WireSeg scene family: {family!r}")
    return SPLITS[family]


def _canonical_path(name):
    path = PurePosixPath(name)
    if (path.is_absolute() or path.as_posix() != name or "\\" in name
            or ".." in path.parts or not path.parts):
        raise ValueError(f"non-canonical WireSeg path: {name!r}")
    return path


def depth_member(file_name, family):
    path = _canonical_path(file_name)
    if (sample_split(family) == "real_test" or path.suffix != ".png"
            or len(path.parts) != 3 or path.parts[0] != family
            or not path.parts[1].removesuffix("_cable").isdigit()
            or not path.parts[1].endswith("_cable")):
        raise ValueError(f"invalid WireSeg depth pairing identity: {file_name!r}")
    return path.with_suffix(".npy").as_posix()


def _compressed_runs(encoded):
    counts = []
    value = shift = 0
    for char in encoded:
        code = ord(char) - 48
        if not 0 <= code <= 63 or shift >= 35:
            raise ValueError("invalid compressed COCO run")
        value |= (code & 31) << shift
        shift += 5
        if code & 32:
            continue
        if code & 16:
            value -= 1 << shift
        if len(counts) > 2:
            value += counts[-2]
        counts.append(value)
        value = shift = 0
    if shift:
        raise ValueError("incomplete compressed COCO run")
    return counts


def decode_coco_rle(rle, shape):
    if rle.get("size") != list(shape):
        raise ValueError("COCO RLE shape must match RGB")
    counts = rle.get("counts")
    if isinstance(counts, str):
        counts = _compressed_runs(counts)
    if (not isinstance(counts, list) or any(type(count) is not int or count < 0 for count in counts)
            or sum(counts) != shape[0] * shape[1]):
        raise ValueError("COCO runs must cover the image exactly")
    return np.repeat(np.arange(len(counts)) % 2 == 1, counts).reshape(shape, order="F")


def _depth_array(payload):
    value = np.load(io.BytesIO(payload), allow_pickle=False)
    if not isinstance(value, np.ndarray) or value.ndim != 2 or value.dtype != np.float32:
        raise ValueError("WireSeg depth must be a two-dimensional float32 NPY")
    return value


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class WireSegSample:
    file_name: str
    scene_family: str
    split: str
    rgb: np.ndarray
    instance_masks: np.ndarray
    depth_m: np.ndarray | None
    depth_valid: np.ndarray | None
    pair_sha256: str | None


def load_wireseg_sample(row, depth_root, *, depth_members=None):
    file_name, family = row["file_name"], row["scene_family"]
    split = sample_split(family)
    _canonical_path(file_name)
    rgb_bytes = row["image"]["bytes"]
    with Image.open(io.BytesIO(rgb_bytes)) as image:
        rgb = np.asarray(image.convert("RGB"))
    shape = rgb.shape[:2]
    encoded = json.loads(row["masks_rle"])
    if not isinstance(encoded, list) or len(encoded) != row["num_instances"]:
        raise ValueError("WireSeg instance count differs from its COCO mask list")
    masks = np.asarray([decode_coco_rle(rle, shape) for rle in encoded], dtype=bool)
    masks = masks.reshape((-1, *shape))
    depth = valid = pair = None
    if split != "real_test":
        root = Path(depth_root).resolve()
        member = depth_member(file_name, family)
        path = (root / member).resolve()
        if not path.is_relative_to(root):
            raise ValueError("WireSeg depth path escapes the data root")
        if (depth_members is None or member in depth_members) and path.is_file():
            payload = path.read_bytes()
            depth = _depth_array(payload)
            if depth.shape != shape:
                raise ValueError("WireSeg RGB/depth shape mismatch")
            valid = np.isfinite(depth) & (depth > 0)
            pair = _sha256(f"{file_name}\n{_sha256(rgb_bytes)}\n{_sha256(payload)}\n".encode())
    return WireSegSample(file_name, family, split, rgb, masks, depth, valid, pair)


def extract_depth_shard(archive, depth_root):
    """Stream checked regular NPY members. Existing byte-identical files allow resume."""
    root = Path(depth_root).resolve()
    names = set()
    with tarfile.open(archive, "r|gz") as bundle:
        for member in bundle:
            name = member.name
            if member.isdir():
                _canonical_path(name.rstrip("/"))
                continue
            path = _canonical_path(name)
            if not member.isfile() or not 0 < member.size < 8 * 1024**2:
                raise ValueError("depth archive requires a bounded regular NPY member")
            if (len(path.parts) != 3 or path.suffix != ".npy"
                    or depth_member(path.with_suffix(".png").as_posix(), path.parts[0]) != name
                    or name in names):
                raise ValueError(f"invalid or duplicate depth archive identity: {name}")
            names.add(name)
            target = (root / name).resolve()
            if not target.is_relative_to(root):
                raise ValueError("depth archive path escapes the data root")
            payload = bundle.extractfile(member).read()
            _depth_array(payload)
            if target.exists():
                if target.read_bytes() != payload:
                    raise ValueError(f"conflicting depth archive identity: {name}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=target.parent, delete=False) as out:
                out.write(payload)
                temporary = Path(out.name)
            temporary.replace(target)
    return sorted(names)


class WireSegPretrainer(nn.Module):
    """Single-image heads with a stride-8 skip over the same GN-ResNet18 backbone."""

    def __init__(self, embed_dim=128, max_instances=32):
        super().__init__()
        self.backbone = ResNet18()
        self.project = nn.Conv2d(512, embed_dim, 1)
        self.skip = nn.Conv2d(128, embed_dim, 1)
        self.instance_head = nn.Conv2d(embed_dim, max_instances, 1)
        self.visible_head = nn.Conv2d(embed_dim, 1, 1)
        self.boundary_head = nn.Conv2d(embed_dim, 1, 1)
        self.depth_head = nn.Conv2d(embed_dim, 1, 1)

    def forward(self, rgb):
        backbone = self.backbone
        detail = backbone.layer2(backbone.layer1(backbone.stem(rgb)))
        context = self.project(backbone.layer4(backbone.layer3(detail)))
        feature = self.skip(detail) + F.interpolate(
            context, size=detail.shape[-2:], mode="bilinear", align_corners=False
        )

        def upsample(value):
            return F.interpolate(value, size=rgb.shape[-2:], mode="bilinear", align_corners=False)

        return {
            "instance": upsample(self.instance_head(feature)),
            "visible": upsample(self.visible_head(feature))[:, 0],
            "boundary": upsample(self.boundary_head(feature))[:, 0],
            "metric_depth": F.softplus(upsample(self.depth_head(feature))[:, 0]),
        }

    def transfer_state(self):
        return {"contract": TRANSFER_CONTRACT, "dataset_revision": WIRESEG_REVISION,
                "backbone": self.backbone.state_dict(), "project": self.project.state_dict()}


def transfer_wireseg_backbone(encoder, artifact):
    """Explicit opt-in transfer. Stereo attention and camera geometry are left untouched."""
    if (not isinstance(encoder, StereoEncoder) or artifact.get("contract") != TRANSFER_CONTRACT
            or artifact.get("dataset_revision") != WIRESEG_REVISION):
        raise ValueError("pinned WireSeg stereo-backbone transfer contract required")
    for module, key in ((encoder.backbone, "backbone"), (encoder.project, "project")):
        current, source = module.state_dict(), artifact[key]
        if current.keys() != source.keys() or any(
                current[name].shape != source[name].shape or current[name].dtype != source[name].dtype
                for name in current):
            raise ValueError("incompatible WireSeg stereo-backbone tensors")
    encoder.backbone.load_state_dict(artifact["backbone"])
    encoder.project.load_state_dict(artifact["project"])


def _mask_loss(logits, target):
    target = target.to(dtype=logits.dtype)
    probability = logits.sigmoid()
    dice = (2 * (probability * target).sum() + 1) / (probability.sum() + target.sum() + 1)
    weight = ((1 - target).sum() / target.sum().clamp_min(1)).clamp(1, 100)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight) + (1 - dice)


def pretraining_losses(prediction, instance_masks, depth_m, depth_valid):
    """Match unordered masks, and supervise metric depth only on known valid pixels."""
    from scipy.optimize import linear_sum_assignment

    logits = prediction["instance"]
    if len(instance_masks) != len(logits):
        raise ValueError("one instance mask set per RGB image required")
    assigned = torch.zeros_like(logits)
    visible = []
    for index, masks in enumerate(instance_masks):
        if masks.ndim != 3 or masks.shape[-2:] != logits.shape[-2:] or len(masks) > logits.shape[1]:
            raise ValueError("instance masks must match the pixel grid and slot budget")
        visible.append(masks.any(0))
        if len(masks):
            with torch.no_grad():
                pred = logits[index].sigmoid().flatten(1)
                truth = masks.flatten(1).to(pred.dtype)
                overlap = pred @ truth.T
                cost = 1 - (2 * overlap + 1) / (pred.sum(1)[:, None] + truth.sum(1)[None] + 1)
                slots, objects = linear_sum_assignment(cost.cpu().numpy())
            assigned[index, slots] = masks[objects].to(assigned.dtype)
    visible = torch.stack(visible)
    support = visible.float().unsqueeze(1)
    edge = F.max_pool2d(support, 3, 1, 1) - (-F.max_pool2d(-support, 3, 1, 1))
    losses = {
        "instance": _mask_loss(logits, assigned),
        "visible": _mask_loss(prediction["visible"], visible),
        "boundary": _mask_loss(prediction["boundary"], edge[:, 0]),
    }
    if depth_m is None and depth_valid is None:
        losses["metric_depth"] = prediction["metric_depth"].new_zeros(())
    else:
        predicted = prediction["metric_depth"]
        if (depth_m.shape != predicted.shape or depth_valid.shape != predicted.shape
                or depth_valid.dtype != torch.bool):
            raise ValueError("metric-depth supervision must align with RGB validity")
        valid = depth_valid & torch.isfinite(depth_m) & (depth_m > 0)
        if valid.any():
            losses["metric_depth"] = F.smooth_l1_loss(predicted[valid], depth_m[valid], beta=0.01)
        else:
            losses["metric_depth"] = predicted.new_zeros(())
    losses["total"] = losses["instance"] + losses["visible"] + 0.25 * losses["boundary"] + losses["metric_depth"]
    return losses
