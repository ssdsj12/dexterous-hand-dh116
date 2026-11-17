"""CPU-only diagnostic inference from mounted-hand and DeformX boundary packets."""

from pathlib import Path

import numpy as np
import torch

from .tabletop_deformable_privileged_teacher import load_deformable_teacher_config
from .tabletop_dual_pinch_builder import BACKEND, BACKEND_SHA
from .tabletop_flexible_measured_common_teacher import MeasuredCommonTeacher
from .tabletop_stereo_contract import digest, file_sha256
from .tabletop_teacher_mounted_wrist_source import validate_mounted_observation

VERSION = "dh116-flexible-native-boundary-inference-cpu-v1"


class NativeBoundaryDiagnosticTeacher(MeasuredCommonTeacher):
    """A separate model identity for inference on the pair-enabled solver's data."""

    def __init__(self, config, base):
        if file_sha256(BACKEND) != BACKEND_SHA:
            raise ValueError("pair-enabled DeformX backend artifact differs")
        super().__init__(config, base)
        self.backend_sha256 = BACKEND_SHA
        contract = self.model_identity
        contract.pop("model_identity_sha256")
        contract.update(
            policy_version=VERSION,
            backend_sha256=BACKEND_SHA,
            checkpoint_namespace=f"deformable_teacher_{base['grasp']}_boundary_diagnostic_cpu_v1",
            source_scope="cpu_diagnostic_no_native_training_admission",
            inference_source_sha256=file_sha256(__file__),
        )
        self._model_identity = dict(contract, model_identity_sha256=digest(contract))

    @classmethod
    def from_project(cls, project, grasp):
        if grasp not in ("pinch", "tripod", "power"):
            raise ValueError("explicit flexible grasp configuration required")
        project = Path(project)
        config, base = load_deformable_teacher_config(
            project / f"configs/dh116_training/privileged_deformable_teacher_{grasp}_v1.json", project
        )
        return cls(config, base)

    @torch.no_grad()
    def pack_boundary(self, preaction):
        """Own one unchanged 371D mask and current material-node packet."""
        if (not isinstance(preaction, dict)
                or preaction.get("native_qualification") is not False
                or preaction.get("teacher_admission") is not False):
            raise ValueError("unqualified pre-action diagnostic packet required")
        if self.log_std.device.type != "cpu":
            raise ValueError("CPU diagnostic inference required")
        common = preaction["common"]
        validate_mounted_observation(common)
        clock = common["identity"]
        boundary = preaction["rod_boundary"]
        tick = clock["rod_tick"]
        if (boundary.get("native_qualified") is not False
                or boundary.get("teacher_admission") is not False
                or boundary.get("run_uuid") != clock["run_uuid"]
                or boundary.get("rod_uid") != clock["rod_uid"]
                or boundary.get("rod_tick") != tick
                or boundary.get("boundary") != ("reset" if tick == 0 else "post_physics")):
            raise ValueError("same-run rod boundary identity differs")
        snapshot = boundary["rod_snapshot"]
        if snapshot.get("source_scope") != "solver_readback_unqualified" or snapshot.get("backend") != "DeformX":
            raise ValueError("DeformX readback with explicit qualification scope required")
        identity = dict(
            env_ids=[clock["env_id"]], episode_uids=[clock["episode_uid"]],
            reset_ids=[clock["reset_id"]], rod_uids=[clock["rod_uid"]],
            physics_step=clock["physics_step"], rod_tick=tick, timestamp_s=clock["timestamp_s"],
        )
        dtype = self.log_std.dtype
        node_fields = ("node_positions", "velocities", "tangent", "curvature", "rest_lengths_m", "deformation")
        snapshot = dict(snapshot)
        for name in node_fields:
            source = snapshot.get(name)
            if isinstance(source, np.ndarray) and source.dtype in (np.float32, np.float64):
                source = torch.from_numpy(source.copy())
            if not isinstance(source, torch.Tensor) or source.dtype not in (torch.float32, torch.float64):
                raise ValueError(f"floating solver readback required for {name}")
            snapshot[name] = source.detach().to(device="cpu", dtype=dtype)
        values = torch.as_tensor(common["values"], dtype=dtype).unsqueeze(0)
        valid = torch.as_tensor(common["valid"]).unsqueeze(0)
        nodes = torch.ones(snapshot["node_positions"].shape[:2], dtype=torch.bool)
        return self.pack_measured(values, valid, snapshot, identity, nodes)
