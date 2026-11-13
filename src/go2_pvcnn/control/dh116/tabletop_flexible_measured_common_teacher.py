"""Independent CPU Teacher using the measured subset of the original 371D ledger.

Unavailable entries remain NaN with explicit validity. The model selects the
same measured indices for every sample; it never feeds imputed rigid-object
fields to the network or accepts the original checkpoint topology.
"""
from pathlib import Path

import torch
from torch import nn

from .tabletop_deformable_privileged_teacher import (
    DeformablePrivilegedTeacher, _check_rod_identity, _features, _mask,
    verify_deformable_teacher_identity,
)
from .tabletop_privileged_teacher import OBSERVATION_SIZE
from .tabletop_stereo_contract import digest, file_sha256


VERSION = "dh116-flexible-measured-common-teacher-cpu-v1"


def common_selection():
    return tuple(range(24)) + tuple(range(78, 88)) + tuple(range(153, 185))


def model_contract(base):
    verify_deformable_teacher_identity(base, base)
    contract = dict(policy_version=VERSION, task_kind=base["task_kind"], grasp=base["grasp"],
        checkpoint_namespace=f"deformable_teacher_{base['grasp']}_measured_common_cpu_v1",
        base_identity_sha256=base["model_identity_sha256"], common_size=OBSERVATION_SIZE,
        common_indices=list(common_selection()), common_missing_value="NaN_with_false_validity",
        common_projection="fixed_original_indices_all_required",
        state_contract_file_sha256=base["state_contract_file_sha256"],
        action_contract_sha256=base["action_contract_sha256"], action_size=base["action_size"],
        rod_schema=base["rod_schema"], rod_features=base["rod_features"],
        backend_sha256=base["backend_sha256"], distribution=base["distribution"],
        model_source_sha256=file_sha256(Path(__file__)),
        native_ready=False, training_reward_qualified=False)
    return dict(contract, model_identity_sha256=digest(contract))


class MeasuredCommonTeacher(DeformablePrivilegedTeacher):
    """Reuse the structured rod encoders and Tanh actor with a fixed 66D projection."""

    def __init__(self, config, base):
        super().__init__(config, base)
        selection = common_selection()
        self.register_buffer("_common_indices", torch.tensor(selection), persistent=False)
        for branch in (self.actor, self.critic):
            branch[0] = nn.Linear(len(selection), branch[0].out_features)
        self._model_identity = model_contract(base)

    def _common(self, ledger, valid):
        if (not isinstance(ledger, torch.Tensor) or ledger.ndim != 2
                or ledger.shape[1] != OBSERVATION_SIZE or not ledger.is_floating_point()
                or not len(ledger) or not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool
                or valid.shape != ledger.shape or valid.device != ledger.device):
            raise ValueError("common ledger requires floating [N,371] and matching boolean validity")
        wanted = torch.zeros_like(valid)
        wanted[:, self._common_indices] = True
        if (not torch.equal(valid, wanted) or not bool(torch.isfinite(ledger[valid]).all())
                or not bool(torch.isnan(ledger[~valid]).all())):
            raise ValueError("common values or validity differ from the fixed measured-field projection")
        return ledger.index_select(1, self._common_indices)

    def pack(self, common, rod_snapshot, common_identity, node_valid):
        raise ValueError("measured common model requires pack_measured and explicit validity")

    def pack_measured(self, ledger, valid, rod_snapshot, common_identity, node_valid):
        selected = self._common(ledger, valid)
        _check_rod_identity(rod_snapshot, common_identity, self.backend_sha256, len(selected))
        return dict(common371=ledger.clone(), common_valid=valid.clone(),
                    rod_nodes=_features(rod_snapshot, selected, node_valid),
                    rod_mask=node_valid.clone())

    def _parts(self, observations):
        if not isinstance(observations, dict) or set(observations) != {
                "common371", "common_valid", "rod_nodes", "rod_mask"}:
            raise ValueError("measured common ledger, explicit validity and rod input required")
        common = self._common(observations["common371"], observations["common_valid"])
        nodes = observations["rod_nodes"]
        if (not isinstance(nodes, torch.Tensor) or nodes.ndim != 3 or len(nodes) != len(common)
                or nodes.shape[-1] != len(self._model_identity["rod_features"])
                or nodes.device != common.device or nodes.dtype != common.dtype):
            raise ValueError("packed rod nodes differ from measured common batch")
        mask = _mask(observations["rod_mask"], nodes.shape[:2], common.device)
        if not bool(torch.isfinite(nodes[mask]).all()):
            raise ValueError("valid packed rod nodes contain unavailable values")
        return common, nodes, mask
