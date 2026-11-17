"""An immutable single-boundary packet for the existing solver state provider."""
from copy import deepcopy

import torch

from .tabletop_dual_hand_rod_runtime import VERSION as RUNTIME_VERSION
from .tabletop_stereo_rod_state import DeformableRodStateProvider, UNITS, VERSION


class _OwnedPacket:
    def __init__(self, packet):
        self.packet = packet

    def read_rod_snapshot(self):
        return self.packet


class RodBoundaryState:
    """Tie the actual reset or measured rod sample to the same current clock.

    The live owner supplies a backend artifact identity from its loaded solver.
    This validates the file and packet only. It does not grant native admission
    or override the original Teacher's frozen backend identity.
    """

    def __init__(self, runtime, backend_artifact, backend_sha256):
        self._runtime = runtime
        self._artifact = backend_artifact
        self._sha256 = backend_sha256

    def capture(self, row):
        row = deepcopy(row)
        clock = self._runtime.clock()
        if (clock["clock_valid"] is not True or row.get("version") != RUNTIME_VERSION
                or any(row.get(key) != value for key, value in clock.items())):
            raise ValueError("rod sample does not match the current runtime clock")
        step = clock["physics_step"]
        tick = clock["rod_tick"]
        reset = tick == 0
        if reset:
            if (row["contact_valid"] is not False or row["distributed_contact"] is not None
                    or row["rod_mesh_contact_pairs"] is not None
                    or row["hand_wrenches_com_world"] is not None or row["table_wrench_world"] is not None):
                raise ValueError("reset contact must be unavailable")
        elif row["contact_valid"] is not True or row["distributed_contact"] is None:
            raise ValueError("measured step requires its solver aggregate contact")

        positions = torch.as_tensor(row["node_positions"]).clone()
        velocities = torch.as_tensor(row["velocities"]).clone()
        rest = torch.as_tensor(row["rest_lengths_m"], dtype=positions.dtype)[None, :].clone()
        if (positions.ndim != 3 or positions.shape[0] != 1 or positions.shape[1] < 3
                or positions.shape[2] != 3 or positions.dtype not in (torch.float32, torch.float64)
                or rest.shape != (1, positions.shape[1] - 1)):
            raise ValueError("one free rod with material nodes and rest segments required")
        edges = torch.diff(positions, dim=1)
        lengths = torch.linalg.vector_norm(edges, dim=-1)
        directions = edges / lengths[..., None]
        tangent = torch.cat((directions[:, :1], directions[:, :-1] + directions[:, 1:],
                             directions[:, -1:]), dim=1)
        tangent = tangent / torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
        curvature = torch.zeros(positions.shape[:2], dtype=positions.dtype)
        curvature[:, 1:-1] = (torch.linalg.vector_norm(torch.diff(directions, dim=1), dim=-1)
                              / (0.5 * (lengths[:, :-1] + lengths[:, 1:])))
        contact = (torch.full_like(positions, float("nan")) if reset else
                   torch.as_tensor(row["distributed_contact"], dtype=positions.dtype).clone())
        identity = dict(env_ids=[clock["env_id"]], episode_uids=[clock["episode_uid"]],
                        reset_ids=[clock["reset_id"]], physics_step=step,
                        timestamp_s=clock["timestamp_s"])
        packet = dict(version=VERSION, coordinate_frame="world", units=dict(UNITS), **identity,
                      node_positions=positions, velocities=velocities, centerline=positions.clone(),
                      tangent=tangent, curvature=curvature, distributed_contact=contact,
                      contact_valid=torch.full(positions.shape[:2], not reset, dtype=torch.bool),
                      rest_lengths_m=rest, deformation=lengths / rest - 1)
        provider = DeformableRodStateProvider(_OwnedPacket(packet), backend="DeformX",
            backend_artifact=self._artifact, backend_sha256=self._sha256)
        state = provider.snapshot(**identity)
        if self._runtime.clock() != clock:
            raise ValueError("rod clock advanced during snapshot")
        state.update(rod_uids=[clock["rod_uid"]], rod_tick=tick)
        return dict(rod_snapshot=state, rod_uid=clock["rod_uid"], rod_tick=tick,
                    boundary="reset" if reset else "post_physics", run_uuid=clock["run_uuid"],
                    native_qualified=False, teacher_admission=False)
