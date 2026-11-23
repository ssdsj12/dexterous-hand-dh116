"""An opt-in 480 Hz barrier for independently owned DH116 physical lanes."""

from copy import deepcopy

import torch

from scripts.verify_dh116_deformx_thumb_contact import step_thumb_physics
from .tabletop_deformable_physical_exchange import FlexiblePhysicalExchange
from .tabletop_surface_contract import validate_actions


def control_group(lanes, actions, *, observe):
    """Collect one diagnostic window, closing the group on any stop."""
    return _control_group(lanes, actions, observe=observe, window_index=None)


def _control_group(lanes, actions, *, observe, window_index):
    """Issue one 60 Hz request per lane, sharing eight global SDK steps."""
    lanes = tuple(lanes)
    validate_actions(actions)
    if (not 2 <= len(lanes) <= 32 or actions.shape != (len(lanes), 24)
            or not callable(observe) or any(not isinstance(lane, FlexiblePhysicalExchange) for lane in lanes)):
        raise ValueError("two to 32 physical lanes, [N,24] actions and a row observer required")
    if any(lane._closed or lane._active for lane in lanes):
        raise RuntimeError("closed or in-progress physical lane")
    sim = lanes[0]._sim
    diagnostic = window_index is None
    control_step = lanes[0]._control if diagnostic else window_index
    if any(lane._sim is not sim or (diagnostic and lane._control != control_step) for lane in lanes):
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
    guard_failed = False
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
                guard = row["guard_failure"]
                guard_failed |= guard is not None
                reason = guard or reason
                if reason is not None:
                    if not isinstance(reason, str) or not reason:
                        raise ValueError("terminal observation must be a nonempty reason or None")
                    terminal[env_id] = reason
            if terminal:
                break
        aborted = bool(terminal) and (diagnostic or guard_failed or substep != 7)
        for env_id, lane in zip(env_ids, lanes, strict=True):
            lane._control += 1
            lane._closed = aborted or env_id in terminal
        report = dict(control_step=control_step, env_ids=env_ids,
                      physical_samples_per_env=substep+1, terminal_reasons=terminal,
                      native_qualification=False, teacher_admission=False)
        if not diagnostic:
            report["aborted"] = aborted
        return report
    except BaseException:
        for lane in lanes:
            lane._closed = True
        raise
    finally:
        for lane in lanes:
            lane._active = False


class TeacherPhysicalGroup:
    """Own shared windows and allow only committed task endings to reset locally."""

    def __init__(self, lanes):
        self._lanes = tuple(lanes)
        self._window_index = 0
        self._done = set()
        self._busy = False
        self._closed = False

    def _close(self):
        self._closed = True
        for lane in self._lanes:
            if isinstance(lane, FlexiblePhysicalExchange):
                lane._closed = True

    def control(self, actions, *, observe, commit):
        """Commit full windows only; the caller settles the trusted PPO sample."""
        if self._closed or self._busy:
            raise RuntimeError("closed or in-progress group")
        if self._done:
            raise RuntimeError("committed terminal lanes require Reset before the next window")
        if not callable(commit):
            raise ValueError("callable full-window commit required")
        self._busy = True
        try:
            report = _control_group(self._lanes, actions, observe=observe,
                                    window_index=self._window_index)
            if report["aborted"]:
                self._close()
                return report
            boundary = [lane._clock(0)[0] for lane in self._lanes]
            commit(deepcopy(report))
            if [lane._clock(0)[0] for lane in self._lanes] != boundary:
                raise ValueError("sample commit changed the owned physical boundary")
            self._window_index += 1
            self._done = set(report["terminal_reasons"])
            return report
        except BaseException:
            self._close()
            raise
        finally:
            self._busy = False

    def reset_done(self, reset_lane):
        """Install fresh, same-owner exchanges at the committed SDK boundary."""
        if self._closed or self._busy or not self._done:
            raise RuntimeError("a complete committed window with terminal lanes is required")
        if not callable(reset_lane):
            raise ValueError("callable lane-local Reset required")

        self._busy = True
        lanes = list(self._lanes)
        try:
            for i, old in enumerate(lanes):
                prior = [lane._clock(0)[0] for lane in lanes]
                env_id = prior[i]["env_id"]
                if env_id not in self._done:
                    continue
                rope = old._rod.rope
                budget_end = old._end
                sdk_time = old._sim.current_time
                replacement = reset_lane(env_id, old)
                self._verify_reset(old, replacement, prior[i])
                if replacement._end != budget_end:
                    raise ValueError("lane Reset changed the owned physics budget")
                if replacement._rod.rope is not rope or old._sim.current_time != sdk_time:
                    raise ValueError("lane Reset replaced the solver or advanced the SDK time")
                fresh = replacement._clock(0)[0]
                for j, lane in enumerate(lanes):
                    if j == i:
                        continue
                    if lane._clock(0)[0] != prior[j]:
                        raise ValueError("Reset changed the clock or identity of another lane")
                    if fresh["episode_uid"] == prior[j]["episode_uid"]:
                        raise ValueError("Reset reused another lane's live episode identity")
                lanes[i] = replacement
            self._lanes = tuple(lanes)
            self._done.clear()
        except BaseException:
            self._lanes = tuple(lanes)
            self._close()
            raise
        finally:
            self._busy = False

    @staticmethod
    def _verify_reset(old, replacement, before):
        if (not isinstance(replacement, FlexiblePhysicalExchange)
                or replacement is old or replacement._sim is not old._sim
                or replacement._rod is not old._rod
                or replacement._closed or replacement._active or replacement._control != 0
                or replacement._reaction is not None
                or bool(replacement._palms.any()) or bool(replacement._angles.any())
                or any(new.hand is not saved.hand or new.env_paths != saved.env_paths
                       for new, saved in zip(replacement._bindings, old._bindings, strict=True))):
            raise ValueError("Reset requires a fresh exchange on the same lane owners")
        after = replacement._clock(0)[0]
        shared = ("run_uuid", "env_id", "rod_uid", "physics_step", "timestamp_s", "solver_time_s")
        if (any(after[key] != before[key] for key in shared)
                or replacement._expected_sdk_step != before["physics_step"]
                or after["episode_uid"] == before["episode_uid"]
                or after["reset_id"] != before["reset_id"] + 1 or after["rod_tick"] != 0):
            raise ValueError("Reset must register a fresh episode without advancing either clock")
