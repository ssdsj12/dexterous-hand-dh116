#!/usr/bin/env python3
"""Reconcile completed timed Pinch windows with immutable native journal rows."""
import argparse
import json
from pathlib import Path
from statistics import median

from scripts.audit_dh116_pinch_recentered_tracking import _checked_bytes

SCOPE = "owner_source_unqualified"
VERSION = "dh116-control-window-wall-v1"
PHASES = ("source_binding", "pre_action", "physical_control", "post_action")


def _windows(journal):
    windows = []
    window = None
    sdk_step = 2
    for index, row in enumerate(journal):
        if type(row["sequence"]) is not int or row["sequence"] != index or row["scope"] != SCOPE:
            raise ValueError("complete original active-source journal sequence required")
        kind, data = row["operation"], row["payload"]
        if kind == "decision":
            if window is not None:
                raise ValueError("unsettled decision cannot be replaced")
            window = {"token": data["collection_token"], "physical": [], "reward": None}
        elif kind in ("delivered_exchange", "settlement", "transition_recorded"):
            if window is None:
                raise ValueError("physical and transition rows require their decision")
            if kind == "delivered_exchange":
                sdk_step += 1
                if data["physics_step"] != sdk_step or window["reward"] is not None:
                    raise ValueError("physical SDK step differs from its ordered control window")
                window["physical"].append(data)
            elif kind == "settlement":
                reward = data["reward"]
                if window["reward"] is not None or reward["token"] != window["token"]:
                    raise ValueError("settlement does not bind the current decision")
                window["reward"] = reward
            else:
                if (window["reward"] is None or data["collection_token"] != window["token"]
                        or data["rollout_length"] != len(windows) + 1):
                    raise ValueError("transition differs from its settled original decision")
                windows.append(window)
                window = None
    if window is not None or not windows:
        raise ValueError("settled and recorded original windows required")
    return windows


def audit_window_wall(trace, journal):
    timers = [row for row in trace if row["phase"] == "control_window_wall"]
    windows = _windows(journal)
    if len(timers) != len(windows):
        raise ValueError("one real timing record is required per original window")
    measured, full, terminal = 0, [], None
    for index, (timer, window) in enumerate(zip(timers, windows)):
        physical, reward = window["physical"], window["reward"]
        times = timer["phase_ns"]
        observer, elapsed = timer["physical_observer_ns"], timer["elapsed_ns"]
        if (timer["version"] != VERSION or timer["scope"] != SCOPE
                or type(timer["control_index"]) is not int or timer["control_index"] != index
                or set(times) != set(PHASES) or any(type(value) is not int or value < 0
                    for value in [*times.values(), observer, elapsed])
                or elapsed != sum(times.values()) or observer > times["physical_control"]
                or timer["transition_recorded"] is not True or timer["quarantined"] is not False
                or timer["native_qualification"] is not False or timer["teacher_admission"] is not False
                or type(timer["sdk_steps_delivered"]) is not int or timer["sdk_steps_delivered"] != len(physical)
                or not 1 <= len(physical) <= 8
                or type(reward["terminated"]) is not bool or type(reward["truncated"]) is not bool
                or timer["terminated"] is not reward["terminated"] or timer["truncated"] is not reward["truncated"]):
            raise ValueError("timing record does not bind the original settled window")
        reason = physical[-1]["guard_failure"]
        if (timer["terminal_reason"] != reason or (reason is not None) is not reward["terminated"]
                or (reward["terminated"] or reward["truncated"]) and index != len(windows) - 1
                or any(row["guard_failure"] is not None for row in physical[:-1])
                or len(physical) != 8 and not reward["terminated"]):
            raise ValueError("termination timing differs from the original protected physics")
        measured += len(physical)
        if reward["terminated"]:
            terminal = physical[-1]
        elif not reward["truncated"]:
            full.append(dict(times, physical_observer=observer,
                             outside_physical_observer=times["physical_control"] - observer, total=elapsed))
    durations = {key: median(row[key] for row in full) for key in full[0]} if full else None
    return dict(recorded_windows=len(windows), complete_windows=len(full), measured_sdk_steps=measured,
                median_complete_ns=durations, terminal_sdk_step=terminal["physics_step"] if terminal else None,
                terminal_reason=terminal["guard_failure"] if terminal else None,
                native_qualification=False, teacher_admission=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for source in ("trace", "journal"):
        parser.add_argument("--" + source, type=Path, required=True)
        parser.add_argument("--" + source + "-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    def read(path, digest):
        return [json.loads(line) for line in _checked_bytes(path, digest).splitlines()]
    result = audit_window_wall(read(args.trace, args.trace_sha256), read(args.journal, args.journal_sha256))
    result.update(trace_sha256=args.trace_sha256, journal_sha256=args.journal_sha256)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, sort_keys=True, indent=2, allow_nan=False)
        out.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
