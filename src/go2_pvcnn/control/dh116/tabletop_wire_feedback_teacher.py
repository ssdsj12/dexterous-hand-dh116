"""Independent CPU Teacher with bounded wire-geometry and force corrections."""
from copy import deepcopy
from pathlib import Path
import re

import torch
from torch import nn

from .tabletop_deformable_privileged_teacher import DeformablePrivilegedTeacher
from .tabletop_deformable_r0_replay import _digest
from .tabletop_native_teacher_model import INPUT_KEYS, NativeContextTeacher, verify_native_teacher_identity
from .tabletop_stereo_contract import digest, file_sha256
from .tabletop_training_reward import WINDOW_SUBSTEPS
from .tabletop_wire_grasp_feedback import PAD_NAMES


VERSION = "dh116-wire-feedback-teacher-cpu-v1"
WIRE_KEYS = ("wire_geometry", "wire_geometry_valid", "wire_reaction", "wire_reaction_valid")
MODEL_INPUTS = (*INPUT_KEYS, *WIRE_KEYS)
_SCHEMA = dict(hands=["left", "right"], pads=list(PAD_NAMES),
               geometry=["x_m", "y_m", "z_m", "tangent_x", "tangent_y", "tangent_z"],
               reaction=["min_normal_n", "mean_normal_n", "force_x_n", "force_y_n", "force_z_n"],
               position_scale_m=.05, force_scale_n=1., actor_latent_correction_limit=.25,
               encoding="asinh_scaled_measurements_then_boolean_validity", frame="decision_palm_local")
_RECORDED_FIELDS = {"position_in_palm_m", "tangent_in_palm", "reaction_force_n", "reaction_valid"}


def _feedback_tensor(observations, key, reference, shape, *, validity=False):
    value = observations[key]
    dtype = torch.bool if validity else reference.dtype
    if (not isinstance(value, torch.Tensor) or value.shape != (len(reference), *shape)
            or value.dtype != dtype or value.device != reference.device):
        raise ValueError(f"wire feedback tensor shape/dtype/device differs for {key}")
    return value


def _observed_values(value, valid, label):
    if not bool(torch.isfinite(value[valid]).all() and torch.isnan(value[~valid]).all()):
        raise ValueError(f"finite measured and NaN unavailable values required for {label}")


def _encode_feedback(observations, common):
    geometry = _feedback_tensor(observations, "wire_geometry", common, (2, 6))
    geometry_valid = _feedback_tensor(observations, "wire_geometry_valid", common, (2,), validity=True)
    reaction = _feedback_tensor(observations, "wire_reaction", common, (2, 6, 5))
    reaction_valid = _feedback_tensor(observations, "wire_reaction_valid", common, (2, 6), validity=True)
    _observed_values(geometry, geometry_valid, "palm geometry")
    _observed_values(reaction, reaction_valid, "palm reaction")
    if bool((reaction_valid & ~geometry_valid[..., None]).any()):
        raise ValueError("a measured reaction requires its valid palm-frame geometry")
    tangent_norm = torch.linalg.vector_norm(geometry[..., 3:][geometry_valid], dim=-1)
    if not torch.allclose(tangent_norm, torch.ones_like(tangent_norm), atol=1e-5, rtol=0.):
        raise ValueError("valid palm geometries require a unit tangent")
    valid_loads = reaction[reaction_valid][:, :2]
    if bool(((valid_loads[:, 0] < 0) | (valid_loads[:, 1] < valid_loads[:, 0])).any()):
        raise ValueError("measured normal loads must be nonnegative and ordered")
    scale = geometry.new_tensor([_SCHEMA["position_scale_m"]]*3 + [1.]*3)
    safe_geometry = torch.where(geometry_valid[..., None], geometry, 0.) / scale
    safe_reaction = torch.where(reaction_valid[..., None], reaction, 0.) / _SCHEMA["force_scale_n"]
    features = torch.cat((safe_geometry.asinh().flatten(1), geometry_valid.to(common).flatten(1),
                          safe_reaction.asinh().flatten(1), reaction_valid.to(common).flatten(1)), dim=1)
    return features, geometry_valid.any(dim=1)


def _recorded_observation(row, clock, pre_action_sha256):
    if not isinstance(pre_action_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", pre_action_sha256):
        raise ValueError("explicit pre-action source SHA-256 required")
    if (not isinstance(clock, dict) or clock.get("clock_valid") is not True
            or type(clock.get("physics_step")) is not int or type(clock.get("rod_tick")) is not int
            or type(clock.get("policy_step")) is not int or min(clock["physics_step"], clock["rod_tick"], clock["policy_step"]) < 0
            or row.get("clock") != clock or row.get("pre_action_sha256") != pre_action_sha256
            or row.get("actor_used") is not False or row.get("teacher_admission") is not False
            or row.get("purpose") not in ("decision", "bootstrap", "bootstrap_truncated")
            or not isinstance(row.get("collection_token"), str) or not row["collection_token"]):
        raise ValueError("same pre-action state and separately recorded wire feedback required")
    observation = row.get("observation")
    if (not isinstance(observation, dict) or set(observation) != _RECORDED_FIELDS
            or row.get("observation_sha256") != _digest(observation)):
        raise ValueError("exact recorded observation fields and SHA-256 required")
    status = row.get("status")
    window = row.get("source_window")
    if status == "no_previous_control_window":
        if clock["policy_step"] != 0 or window is not None or row["purpose"] != "decision":
            raise ValueError("missing previous control must be from the first decision")
    elif (status not in ("measured", "reaction_unavailable") or clock["policy_step"] == 0
            or window != dict(first_sdk_step=clock["physics_step"]-WINDOW_SUBSTEPS+1,
                              end_sdk_step=clock["physics_step"], end_rod_tick=clock["rod_tick"],
                              control_index=clock["policy_step"]-1)):
        raise ValueError("complete preceding control interval required")
    return observation, status


class WireFeedbackTeacher(DeformablePrivilegedTeacher):
    """Use a private copy of the established Teacher and learn a 24D correction."""

    def __init__(self, base):
        if type(base) is not NativeContextTeacher or any(p.device.type != "cpu" for p in base.parameters()):
            raise ValueError("exact CPU NativeContextTeacher base required")
        old_identity = base.model_identity
        verify_native_teacher_identity(old_identity, old_identity)
        nn.Module.__init__(self)
        self.base = deepcopy(base)
        self.actor_feedback, self.critic_feedback = self._branch(24), self._branch(1)
        self.actor_feedback.to(self.base.log_std)
        self.critic_feedback.to(self.base.log_std)
        identity = dict(policy_version=VERSION, checkpoint_namespace=f"wire_feedback_{old_identity['grasp']}_cpu_v1",
                        grasp=old_identity["grasp"], base_model=old_identity, input_keys=list(MODEL_INPUTS),
                        action_size=24, distribution=old_identity["distribution"], input_schema=deepcopy(_SCHEMA),
                        input_schema_sha256=digest(dict(keys=list(MODEL_INPUTS), fields=_SCHEMA)),
                        implementation_sha256=file_sha256(Path(__file__)), native_ready=False)
        self._model_identity = dict(identity, model_identity_sha256=digest(identity))

    @staticmethod
    def _branch(width):
        features = 2 * (6 + 1 + 6 * (5 + 1))
        head = nn.Linear(64, width)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        return nn.Sequential(nn.Linear(features, 64), nn.ELU(), head)

    @property
    def log_std(self):
        return self.base.log_std

    def pack(self, *args, **kwargs):
        raise ValueError("pack the original observation first, then attach separately recorded wire feedback")

    def missing_feedback(self, base_observation):
        """An explicit whole-feedback ablation, with unavailable raw values."""
        common, _, _ = self.base._parts(base_observation)
        batch = len(common)
        return dict(**{key: tensor.clone() for key, tensor in base_observation.items()},
                    wire_geometry=common.new_full((batch, 2, 6), float("nan")),
                    wire_geometry_valid=torch.zeros((batch, 2), dtype=torch.bool, device=common.device),
                    wire_reaction=common.new_full((batch, 2, 6, 5), float("nan")),
                    wire_reaction_valid=torch.zeros((batch, 2, 6), dtype=torch.bool, device=common.device))

    def pack_recorded_feedback(self, base_observation, rows, *, base_observation_sha256,
                               clocks, pre_action_sha256s):
        """Pack replay feedback against separately bound original observation bytes and identities."""
        if _digest(base_observation) != base_observation_sha256:
            raise ValueError("base observation SHA-256 differs from the bound source")
        result = self.missing_feedback(base_observation)
        if len(rows) != len(result["common371"]) or len(clocks) != len(rows) or len(pre_action_sha256s) != len(rows):
            raise ValueError("wire feedback must identify every observation in the batch")
        geometry = result["wire_geometry"]
        reaction = result["wire_reaction"]
        for index, (row, clock, pre_sha) in enumerate(zip(rows, clocks, pre_action_sha256s, strict=True)):
            observation, status = _recorded_observation(row, clock, pre_sha)
            xyz = geometry.new_tensor(observation["position_in_palm_m"])
            tangent = geometry.new_tensor(observation["tangent_in_palm"])
            force = reaction.new_tensor(observation["reaction_force_n"])
            valid = torch.as_tensor(observation["reaction_valid"], device=reaction.device)
            if (xyz.shape != (2, 3) or tangent.shape != (2, 3) or force.shape != (2, 6, 5)
                    or valid.shape != (2, 6) or valid.dtype != torch.bool):
                raise ValueError("recorded geometry, force or validity shape differs")
            if bool(valid.any()) != (status == "measured") or not bool((force[~valid] == 0).all()):
                raise ValueError("recorded measurement status or masked force placeholders differ")
            geometry[index] = torch.cat((xyz, tangent), dim=1)
            result["wire_geometry_valid"][index] = True
            reaction[index] = torch.where(valid[..., None], force, float("nan"))
            result["wire_reaction_valid"][index] = valid
        self._parts(result)
        return result

    @staticmethod
    def _base_inputs(observations):
        if not isinstance(observations, dict) or set(observations) != set(MODEL_INPUTS):
            raise ValueError("exact original and wire-feedback input keys required")
        return {key: observations[key] for key in INPUT_KEYS}

    def _parts(self, observations):
        parts = self.base._parts(self._base_inputs(observations))
        _encode_feedback(observations, parts[0])
        return parts

    def latent_mean(self, observations):
        base = self._base_inputs(observations)
        mean = self.base.latent_mean(base)
        features, active = _encode_feedback(observations, base["common371"])
        delta = _SCHEMA["actor_latent_correction_limit"] * self.actor_feedback(features).tanh()
        return torch.where(active[:, None], mean+delta, mean)

    def value(self, observations):
        base = self._base_inputs(observations)
        value = self.base.value(base)
        features, active = _encode_feedback(observations, base["common371"])
        return torch.where(active, value+self.critic_feedback(features).squeeze(-1), value)

    def get_extra_state(self):
        return self.model_identity

    def set_extra_state(self, state):
        if state != self._model_identity:
            raise ValueError("independent wire-feedback Teacher identity differs")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.set_extra_state(state_dict.get("_extra_state"))
        self.base.set_extra_state(state_dict.get("base._extra_state"))
        if not strict or state_dict.keys() != self.state_dict().keys():
            raise ValueError("complete wire-feedback Teacher weights required")
        for name, expected in self.state_dict().items():
            if isinstance(expected, torch.Tensor):
                supplied = state_dict[name]
                if (not isinstance(supplied, torch.Tensor) or supplied.shape != expected.shape
                        or supplied.dtype != expected.dtype or supplied.device != expected.device
                        or not bool(torch.isfinite(supplied).all())):
                    raise ValueError(f"invalid wire-feedback Teacher weight {name}")
        return super().load_state_dict(state_dict, strict=True, assign=assign)
