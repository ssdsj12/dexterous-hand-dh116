"""Join independently validated Teacher observations at one shared SDK boundary."""

import torch

from .tabletop_native_teacher_context import TaskContextPacket
from .tabletop_native_teacher_model import INPUT_KEYS, NativeContextTeacher


def pack_teacher_group(model, lane_packets, *, backend_profile):
    """Pack one observation per environment in ascending env_id order."""
    packets = tuple(lane_packets)
    if type(model) is not NativeContextTeacher or not 2 <= len(packets) <= 32:
        raise ValueError("one native Teacher and two to 32 source packets required")
    contexts = [packet[3] for packet in packets]
    if any(not isinstance(context, TaskContextPacket) or len(context.metadata.identities) != 1
           for context in contexts):
        raise ValueError("exactly one owning task context per environment required")
    clocks = [context.metadata.identities[0] for context in contexts]
    env_ids = [clock.env_id for clock in clocks]
    if (env_ids != sorted(set(env_ids))
            or len({clock.episode_uid for clock in clocks}) != len(clocks)):
        raise ValueError("ordered distinct environment and episode identities required")
    if len({(clock.run_uuid, clock.physics_step, clock.timestamp_s) for clock in clocks}) != 1:
        raise ValueError("one same-time, same-run SDK boundary required")
    scopes = {context.metadata.scope for context in contexts}
    if len(scopes) != 1:
        raise ValueError("CPU fixtures and live ownership cannot share a policy batch")
    if scopes != {"cpu_fixture"}:
        owners = [packet[2].get("source_instance_id") for packet in packets]
        if any(type(owner) is not int or owner <= 0 for owner in owners) or len(set(owners)) != len(owners):
            raise ValueError("every environment must bind its own actual rod solver")
    # Validate raw SDK/rod identity, validity and fixed node data per environment.
    packed = [model.pack_native_inputs(*packet, backend_profile=backend_profile) for packet in packets]
    return {key: torch.cat([row[key] for row in packed], dim=0) for key in INPUT_KEYS}
