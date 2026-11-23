"""An opt-in 480 Hz barrier for independently owned DH116 physical lanes."""

from copy import deepcopy

import torch

from scripts.verify_dh116_deformx_thumb_contact import step_thumb_physics
from .tabletop_deformable_physical_exchange import FlexiblePhysicalExchange
from .tabletop_surface_contract import validate_actions


def control_group(lanes, actions, *, observe):
    """Issue one 60 Hz request per lane, sharing eight global SDK steps.

    The caller supplies live per-lane exchanges and a durable sink for raw rows.
    Until individual episode recovery is bound, a terminal lane closes the whole
    group at the completed barrier. Rows remain available for diagnosis.
    """
    lanes = tuple(lanes)
    validate_actions(actions)
    if (not 2 <= len(lanes) <= 32 or actions.shape != (len(lanes), 24)
            or not callable(observe) or any(not isinstance(lane, FlexiblePhysicalExchange) for lane in lanes)):
        raise ValueError("two to 32 physical lanes, [N,24] actions and a row observer required")
    if any(lane._closed or lane._active for lane in lanes):
        raise RuntimeError("closed or in-progress physical lane")
    sim = lanes[0]._sim
    control_step = lanes[0]._control
    if any(lane._sim is not sim or lane._control != control_step for lane in lanes):
        raise ValueError("one SDK owner and one shared policy boundary required")

    rows = [lane._clock(0)[0] for lane in lanes]
    env_ids = [row["env_id"] for row in rows]
    if (env_ids != sorted(set(env_ids)) or len({row["episode_uid"] for row in rows}) != len(lanes)
            or len({row["run_uuid"] for row in rows}) != 1):
        raise ValueError("ordered distinct environment and episode identities within one run required")
    if len({id(lane._rod.rope) for lane in lanes}) != len(lanes):
        raise ValueError("distinct per-environment solver owners required")
    bindings = [binding for lane in lanes for binding in lane._bindings]
    if (len({id(binding.hand) for binding in bindings}) != len(bindings)
            or len({id(binding.hand.root_physx_view) for binding in bindings}) != len(bindings)
            or len({path for binding in bindings for path in binding.env_paths.values()}) != len(bindings)):
        raise ValueError("distinct original articulation and SDK view identities required")

    budget_end = min(lane._end for lane in lanes)
    if sim.current_time_step_index + 8 > budget_end:
        raise RuntimeError("whole control interval exceeds a lane physics budget")
    if any(lane._expected_sdk_step != sim.current_time_step_index for lane in lanes):
        raise ValueError("every rod must own the current SDK boundary")
    actions = actions.detach().cpu().to(dtype=torch.float64).clone()

    terminal = {}
    for lane in lanes:
        lane._active = True
    try:
        for substep in range(8):
            expected = sim.current_time_step_index
            pending = [lane._prepare_substep(action[None], substep)
                       for lane, action in zip(lanes, actions, strict=True)]
            sdk_step = step_thumb_physics(sim, expected, budget_end=budget_end)
            for env_id, lane, prepared in zip(env_ids, lanes, pending, strict=True):
                row = lane._complete_substep(prepared, sdk_step)
                reason = observe(deepcopy(row))
                reason = row["guard_failure"] or reason
                if reason is not None:
                    if not isinstance(reason, str) or not reason:
                        raise ValueError("terminal observation must be a nonempty reason or None")
                    terminal[env_id] = reason
            if terminal:
                break
        for lane in lanes:
            lane._control += 1
            lane._closed = bool(terminal)
        return dict(control_step=control_step, env_ids=env_ids,
                    physical_samples_per_env=substep+1, terminal_reasons=terminal,
                    native_qualification=False, teacher_admission=False)
    except BaseException:
        for lane in lanes:
            lane._closed = True
        raise
    finally:
        for lane in lanes:
            lane._active = False
