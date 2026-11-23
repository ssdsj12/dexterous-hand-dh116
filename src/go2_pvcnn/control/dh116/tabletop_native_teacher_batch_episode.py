"""Native batch episode extension that preserves the historical single-lane source."""
from copy import deepcopy

from .tabletop_deformable_dual_hand_episode import DualHandPhysicalEpisode, CLEARANCE_M, SEED_SOURCE, SEED_VERSION


class NativeBatchPhysicalEpisode(DualHandPhysicalEpisode):
    """Reuse original readbacks in a shared step barrier."""

    def install_batch_reset(self, exchange, receipt):
        """Adopt a lane's completed native mesh rebuild on the existing stage."""
        if (self._state != "new" or exchange._sim is not self._sim or exchange._rod is not self._rod
                or exchange._bindings != self._bindings or exchange._end != self._end
                or receipt["after_clock"] != self._rod.clock() or receipt["status"] != "complete"
                or receipt["batch_mesh_rebuild"]["source_instance_id"] != id(self._rod.rope)):
            raise ValueError("fresh verified lane Reset on the same owners required")
        self._exchange = exchange
        self._rod_packet = self._boundary.capture(receipt["rod_reset"])
        self._state_seed = dict(version=SEED_VERSION, source=SEED_SOURCE,
            identity=self._read_state_clock(), authored_hand_roots_w=receipt["seed"]["hand_roots"],
            environment_origin_w=receipt["seed"]["origin"], root_xy=receipt["seed"]["root_xy"],
            root_seed_w=receipt["seed"]["authored_root_seed_w"])
        evidence = dict(kind="reset", reset=receipt, rod_boundary=deepcopy(self._rod_packet),
            clearance_m=exchange._read_geometry()["clearance_m"], clearance_floor_m=CLEARANCE_M,
            clearance_passed=True, native_qualification=False, teacher_admission=False)
        self._record(deepcopy(evidence))
        self._state = "active"
        return evidence

    def record_physical_row(self, row):
        """Persist the exact measured row for both direct and lockstep steps."""
        packet = self._boundary.capture(row["rod"])
        event = dict(kind="physical", exchange=row, rod_boundary=deepcopy(packet),
                     native_qualification=False, teacher_admission=False)
        self._record(event)
        self._rod_packet = packet
        return event

    def finish_group_window(self, *, terminal):
        if self._state != "active" or self._exchange._control != self._control_index + 1:
            raise ValueError("only the completed owning exchange can advance the episode window")
        self._control_index += 1
        self._state = "closed" if terminal else "active"
