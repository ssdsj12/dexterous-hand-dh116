"""Record simulation and host timing at the actual Teacher action boundary."""
import math
from statistics import median
from time import monotonic_ns


class ActionBoundary:
    def __init__(self):
        self.begin()

    def begin(self):
        self.issued = self.integrated = self.applied = None

    def __call__(self, phase, sim, act):
        wall = monotonic_ns()
        stamp = dict(sdk_step=int(sim.current_time_step_index),
                     sim_time_s=float(sim.current_time), wall_ns=wall)
        if phase == "issued" and self.issued is None:
            self.issued = stamp
            self.applied = act
        elif phase == "integrated" and self.issued is not None and self.integrated is None:
            self.integrated = stamp
        else:
            raise ValueError("unexpected Teacher action hook order")


def sync_row(receipt, issued, integrated, *, capture_start_ns, rgb_ready_ns, final_step,
             actor_action, applied_action):
    identity, render = receipt["identity"], receipt["render_timing"]
    left, right = render["raw_left_clock"], render["raw_right_clock"]
    if left != right:
        raise ValueError("left and right renderer clocks differ")
    step, phys_time = identity["sdk_step"], identity["sdk_timestamp_s"]
    if (not issued or not integrated or issued["sdk_step"] != step
            or integrated["sdk_step"] != step + 1 or final_step != step + 8
            or not math.isclose(integrated["sim_time_s"] - phys_time, 1/480, rel_tol=0, abs_tol=1e-6)):
        raise ValueError("action was not observed at the expected SDK boundaries")
    wall = (capture_start_ns, rgb_ready_ns, issued["wall_ns"], integrated["wall_ns"])
    if any(type(t) is not int or t < 0 for t in wall) or tuple(sorted(wall)) != wall:
        raise ValueError("invalid monotonic action and RGB timing")
    for action in (actor_action, applied_action):
        if len(action) != 24 or not all(math.isfinite(v) for v in action):
            raise ValueError("finite 24D action required")
    if any(actual != min(1., max(-1., requested)) for requested, actual in zip(actor_action, applied_action)):
        raise ValueError("inference and controller inputs differ beyond the frozen clip")
    rgb_time, act_time = left["simulation_time"], issued["sim_time_s"]
    if not all(math.isfinite(t) and t >= 0 for t in (phys_time, rgb_time, act_time)):
        raise ValueError("invalid native simulation timestamps")
    epoch = left["render_frame_number"]
    return dict(identity=identity, capture_id=render["capture_id"], renderer_epoch=epoch,
                raw_render_clocks=dict(left=left,right=right), T_rgb_s=rgb_time,
                T_phys_s=phys_time, T_act_s=act_time, T_act_first_integrated_s=integrated["sim_time_s"],
                post_control_sdk_step=final_step, rgb_phys_delta_ms=(rgb_time-phys_time)*1000,
                act_phys_delta_ms=(act_time-phys_time)*1000, capture_start_wall_ns=capture_start_ns,
                rgb_ready_wall_ns=rgb_ready_ns, act_issued_wall_ns=issued["wall_ns"],
                act_integrated_wall_ns=integrated["wall_ns"], actor_action_24=actor_action,
                controller_input_24=applied_action)


def _stats(values):
    values = sorted(values)
    if not values:
        return None
    return dict(min=values[0], median=median(values),
                p95=values[math.ceil(.95*len(values))-1], max=values[-1])


def timing_summary(rows):
    intervals = []
    for before, after in zip(rows, rows[1:]):
        a, b = before["identity"], after["identity"]
        if (b["policy_tick"] != a["policy_tick"] + 1 or b["sdk_step"] != before["post_control_sdk_step"]
                or after["renderer_epoch"] <= before["renderer_epoch"]
                or after["capture_id"] != before["capture_id"] + 1
                or after["capture_start_wall_ns"] < before["act_integrated_wall_ns"]):
            raise ValueError("non-contiguous action/camera sequence")
        intervals.append((after["rgb_ready_wall_ns"]-before["rgb_ready_wall_ns"])/1e6)
    skew = [abs(row["rgb_phys_delta_ms"]) for row in rows]
    offsets = [row["act_phys_delta_ms"] for row in rows]
    spread = max(offsets)-min(offsets) if len(offsets) >= 2 else None
    def latency(start, end):
        return _stats([(row[end]-row[start])/1e6 for row in rows])
    return dict(frames=len(rows), rgb_phys_under_2ms=all(v < 2. for v in skew) if rows else None,
                rgb_phys_abs_ms=_stats(skew), action_sim_offset_range_ms=spread,
                action_sim_offset_constant=spread <= .001 if spread is not None else None,
                action_same_sdk_boundary=all(abs(v) <= .001 for v in offsets) if offsets else None,
                frame_wall_interval_ms=_stats(intervals),
                frame_wall_interval_range_ms=max(intervals)-min(intervals) if intervals else None,
                reprojection_px=None,
                capture_wall_latency_ms=latency("capture_start_wall_ns","rgb_ready_wall_ns"),
                rgb_to_issue_wall_ms=latency("rgb_ready_wall_ns","act_issued_wall_ns"),
                rgb_to_first_integrated_wall_ms=latency("rgb_ready_wall_ns","act_integrated_wall_ns"))
