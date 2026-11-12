"""Separate, CPU-testable DH116 flexible Teacher model contract."""

import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn.functional import pad

from .tabletop_deformable_grasp_task import GRASPS, load_flexible_grasp_task
from .tabletop_privileged_teacher import ACTION_SIZE, OBSERVATION_SIZE, PrivilegedTeacher, _observation
from .tabletop_stereo_contract import digest, file_sha256
from .tabletop_stereo_rod_state import VERSION as ROD_VERSION, _validate_clock


VERSION = "dh116-deformable-privileged-teacher-tanh-v1"
CONFIG_VERSION = "dh116-deformable-privileged-teacher-config-v1"
ROD_FEATURES = (
    "position_world_x_m", "position_world_y_m", "position_world_z_m",
    "velocity_world_x_m_s", "velocity_world_y_m_s", "velocity_world_z_m_s",
    "tangent_world_x", "tangent_world_y", "tangent_world_z", "curvature_m_inv",
    "normalized_rest_material_station", "adjacent_segment_strain", "adjacent_segment_strain_valid",
)
_TASK_CONFIG = "configs/dh116_tasks/flexible_dual_hand_grasp_v1.json"
_SAMPLE_IDENTITY = ("env_ids", "episode_uids", "reset_ids", "rod_uids", "physics_step", "rod_tick", "timestamp_s")
_NATIVE_PREREQUISITES = (
    "qualified_same_rod_dual_hand_contact", "qualified_reset_and_371d_common",
    "qualified_flexible_r0", "approved_bounded_native_budget",
)


def load_deformable_teacher_config(path, project_root):
    """Bind one grasp-specific CPU model to the immutable task and original contracts."""
    raw = Path(path).read_bytes()
    config = json.loads(raw)
    expected_keys = {"version", "task_kind", "grasp", "task_config_sha256", "checkpoint_namespace",
                     "model", "reward_profile", "use_cameras", "expert_initialization",
                     "native_ready", "native_prerequisites"}
    if (not isinstance(config, dict) or set(config) != expected_keys
            or config["version"] != CONFIG_VERSION or config["task_kind"] != "deformable_rod"
            or config["grasp"] not in GRASPS or config["use_cameras"] is not False
            or config["expert_initialization"] != "optional" or config["native_ready"] is not False
            or config["reward_profile"] != "flexible_r0"
            or config["native_prerequisites"] != list(_NATIVE_PREREQUISITES)):
        raise ValueError("expected camera-free, native-disabled flexible Teacher configuration")
    grasp = config["grasp"]
    if config["checkpoint_namespace"] != f"deformable_teacher_{grasp}_v1":
        raise ValueError("independent grasp-specific checkpoint namespace required")
    model = config["model"]
    model_fields = {"common_size": OBSERVATION_SIZE, "action_size": ACTION_SIZE, "hidden": [256, 256],
                    "rod_feature_size": len(ROD_FEATURES), "rod_hidden": 64}
    if (not isinstance(model, dict) or set(model) != set(model_fields) | {"initial_std"}
            or any(model[key] != value for key, value in model_fields.items())
            or type(model["initial_std"]) not in (int, float)
            or not .001 <= model["initial_std"] <= .3):
        raise ValueError("expected original common371/action24 and independent masked rod encoder")
    project = Path(project_root).resolve()
    task_path = project / _TASK_CONFIG
    if file_sha256(task_path) != config["task_config_sha256"]:
        raise ValueError("flexible Teacher task config SHA256 mismatch")
    task = load_flexible_grasp_task(task_path, project)
    identity = {
        "policy_version": VERSION, "task_kind": "deformable_rod", "grasp": grasp,
        "checkpoint_namespace": config["checkpoint_namespace"],
        "config_sha256": hashlib.sha256(raw).hexdigest(), "config_semantic_sha256": digest(config),
        "task_config_sha256": config["task_config_sha256"], "model_source_sha256": file_sha256(Path(__file__)),
        "dependency_source_sha256": {name: file_sha256(Path(__file__).with_name(name + ".py"))
                                     for name in ("tabletop_privileged_teacher", "tabletop_squashed_gaussian",
                                                  "tabletop_stereo_rod_state")},
        "common_size": OBSERVATION_SIZE, "action_size": ACTION_SIZE,
        "action_contract_sha256": task["contracts"]["action"]["contract_sha256"],
        "state_contract_file_sha256": task["sources"]["state"]["sha256"],
        "rod_schema": ROD_VERSION, "rod_features": list(ROD_FEATURES),
        "backend_sha256": task["backend"]["extension_sha256"], "distribution": "tanh_latent_gaussian",
        "native_ready": False,
    }
    identity["model_identity_sha256"] = digest(identity)
    return config, identity


def verify_deformable_teacher_identity(observed, expected):
    """Reject old model contracts and identities from another grasp/configuration."""
    if (not isinstance(expected, dict) or expected.get("policy_version") != VERSION
            or expected.get("native_ready") is not False
            or expected.get("model_identity_sha256") != digest({
                key: value for key, value in expected.items() if key != "model_identity_sha256"
            }) or observed != expected):
        raise ValueError("flexible Teacher model identity differs")


def _check_rod_identity(snapshot, common_identity, backend_sha, batch):
    if (not isinstance(snapshot, dict) or snapshot.get("version") != ROD_VERSION
            or snapshot.get("task_kind") != "deformable_rod" or snapshot.get("coordinate_frame") != "world"
            or snapshot.get("backend_sha256") != backend_sha
            or snapshot.get("source_scope") not in ("solver_readback_unqualified", "cpu_fixture")):
        raise ValueError("expected a versioned world-frame DeformX rod schema and backend identity")
    if not isinstance(common_identity, dict) or set(common_identity) != set(_SAMPLE_IDENTITY):
        raise ValueError("common state requires complete same-clock rod identity")
    for item in (common_identity, snapshot):
        try:
            _validate_clock(*(item[key] for key in ("env_ids", "episode_uids", "reset_ids", "physics_step",
                                                   "timestamp_s")))
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError("invalid physics sample identity") from error
        ids = item.get("rod_uids")
        if (len(item["env_ids"]) != batch or not isinstance(ids, (list, tuple)) or len(ids) != batch
                or any(not isinstance(uid, str) or not uid for uid in ids)):
            raise ValueError("one identified rod per environment required")
        if type(item.get("rod_tick")) is not int or item["rod_tick"] < 0:
            raise ValueError("physics sample identity requires an explicit rod tick")
    for key in _SAMPLE_IDENTITY:
        left, right = common_identity[key], snapshot[key]
        if key == "timestamp_s":
            equal = math.isclose(left, right, abs_tol=1e-9, rel_tol=0.)
        elif isinstance(left, (list, tuple)):
            equal = tuple(left) == tuple(right)
        else:
            equal = left == right
        if not equal:
            raise ValueError(f"common and rod physics sample identity differs at {key}")


def _mask(mask, shape, device):
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != shape or mask.device != device:
        raise ValueError("explicit boolean rod node mask shape/device differs")
    if bool((mask.sum(dim=1) < 3).any()):
        raise ValueError("each rod requires at least three valid material nodes")
    return mask


def _rod_tensor(snapshot, name, reference, shape):
    value = snapshot.get(name)
    if (not isinstance(value, torch.Tensor) or value.shape != shape
            or value.dtype != reference.dtype or value.device != reference.device):
        raise ValueError(f"rod {name} tensor shape/dtype/device differs")
    return value


def _features(snapshot, common, mask):
    nodes = snapshot.get("node_positions")
    if not isinstance(nodes, torch.Tensor) or nodes.ndim != 3:
        raise ValueError("rod node_positions shape must be [batch,nodes,3]")
    batch, count = common.shape[0], nodes.shape[1]
    mask = _mask(mask, (batch, count), common.device)
    shape = (batch, count, 3)
    fields = []
    for name in ("node_positions", "velocities", "tangent"):
        value = _rod_tensor(snapshot, name, common, shape)
        if not bool(torch.isfinite(value[mask]).all()):
            raise ValueError(f"valid rod {name} contains unavailable values")
        if name == "tangent":
            norms = torch.linalg.vector_norm(value[mask], dim=-1)
            if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=0.):
                raise ValueError("valid rod nodes require a unit tangent")
        fields.append(torch.where(mask[..., None], value, 0.))
    curvature = _rod_tensor(snapshot, "curvature", common, (batch, count))
    if not bool((torch.isfinite(curvature[mask]) & (curvature[mask] >= 0)).all()):
        raise ValueError("valid curvature must be finite and nonnegative")
    lengths = _rod_tensor(snapshot, "rest_lengths_m", common, (batch, count - 1))
    if not bool((torch.isfinite(lengths) & (lengths > 0)).all()):
        raise ValueError("rest material lengths require finite positive segment geometry")
    station = pad(lengths.cumsum(-1), (1, 0)) / lengths.sum(-1, keepdim=True)
    strain = _rod_tensor(snapshot, "deformation", common, (batch, count - 1))
    segment_valid = mask[:, :-1] & mask[:, 1:]
    if not bool(torch.isfinite(strain[segment_valid]).all()):
        raise ValueError("valid adjacent segment strain is unavailable")
    observed_strain = torch.where(segment_valid, strain, 0.)
    weights = segment_valid.to(dtype=common.dtype)
    count_valid = pad(weights, (1, 0)) + pad(weights, (0, 1))
    average_strain = (pad(observed_strain, (1, 0)) + pad(observed_strain, (0, 1))) / count_valid.clamp_min(1.)
    scalars = torch.stack((torch.where(mask, curvature, 0.), station,
                           average_strain, (count_valid > 0).to(dtype=common.dtype)), dim=-1)
    return torch.where(mask[..., None], torch.cat((*fields, scalars), dim=-1), 0.)


class _MaterialEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.per_node = nn.Sequential(nn.Linear(len(ROD_FEATURES), width), nn.ELU(),
                                      nn.Linear(width, width), nn.ELU())

    def forward(self, nodes, mask):
        safe = torch.where(mask[..., None], nodes, 0.)
        encoded = self.per_node(safe)
        active = mask[..., None]
        mean = torch.where(active, encoded, 0.).sum(1) / mask.sum(1, keepdim=True)
        maximum = encoded.masked_fill(~active, -torch.inf).amax(1)
        return torch.cat((mean, maximum), dim=-1)


class DeformablePrivilegedTeacher(PrivilegedTeacher):
    """Reuse the existing Tanh sampling and likelihood with separate rod encoders."""

    def __init__(self, config, identity):
        super().__init__(initial_std=config["model"]["initial_std"])
        verify_deformable_teacher_identity(identity, identity)
        if identity["config_semantic_sha256"] != digest(config):
            raise ValueError("flexible Teacher config and model identity differ")
        self.backend_sha256 = identity["backend_sha256"]
        width = config["model"]["rod_hidden"]
        self.actor = self.actor[:-1]
        self.critic = self.critic[:-1]
        self.actor_rod = _MaterialEncoder(width)
        self.critic_rod = _MaterialEncoder(width)
        self.actor_head = nn.Linear(256 + 2 * width, ACTION_SIZE)
        self.critic_head = nn.Linear(256 + 2 * width, 1)

    def pack(self, common, rod_snapshot, common_identity, node_valid):
        common = _observation(common)
        _check_rod_identity(rod_snapshot, common_identity, self.backend_sha256, len(common))
        features = _features(rod_snapshot, common, node_valid)
        return dict(common=common.clone(), rod_nodes=features, rod_mask=node_valid.clone())

    def _parts(self, observations):
        if not isinstance(observations, dict) or set(observations) != {"common", "rod_nodes", "rod_mask"}:
            raise ValueError("flexible Teacher requires common371 and masked rod node inputs")
        common = _observation(observations["common"])
        nodes = observations["rod_nodes"]
        if (not isinstance(nodes, torch.Tensor) or nodes.ndim != 3 or nodes.shape[0] != len(common)
                or nodes.shape[-1] != len(ROD_FEATURES) or nodes.dtype != common.dtype
                or nodes.device != common.device):
            raise ValueError("packed rod node shape/dtype/device differs")
        mask = _mask(observations["rod_mask"], nodes.shape[:2], common.device)
        if not bool(torch.isfinite(nodes[mask]).all()):
            raise ValueError("valid packed rod nodes contain unavailable values")
        return common, nodes, mask

    def latent_mean(self, observations):
        common, nodes, mask = self._parts(observations)
        return self.actor_head(torch.cat((self.actor(common), self.actor_rod(nodes, mask)), dim=-1))

    def value(self, observations):
        common, nodes, mask = self._parts(observations)
        return self.critic_head(torch.cat((self.critic(common), self.critic_rod(nodes, mask)), dim=-1)).squeeze(-1)
