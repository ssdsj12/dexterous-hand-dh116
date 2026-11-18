"""Streaming, CPU-only diagnostics for measured grasp contacts.

This module only reduces supplied measurements: it never controls physics or
changes the native success criterion. All lengths/times/loads/torques are SI.
A pad normal is the *reduced_native_force_direction*, not a manifold normal;
per-point native positions and normals remain in the streamed raw_contacts.
"""
from __future__ import annotations

from copy import deepcopy
import math

SCHEMA_VERSION = "grasp_contact_diagnostics_v1"
HANDS = ("left", "right")
DIGITS = ("thumb", "index", "middle", "ring", "little")
STYLE_MASKS = {
    "pinch_thumb_index": ("thumb", "index"),
    "tripod_thumb_index_middle": ("thumb", "index", "middle"),
}
REASONS = ("NO_CONTACT", "ONE_SIDE_CONTACT", "WRONG_NORMAL", "CONTACT_LOSS",
           "NON_OPPOSED", "TORQUE_LIMITED", "SLIP")
TORQUE_SOURCE = "implicit_actuator_estimate_not_hardware_measurement"
CONTRACT = dict(
    contact_threshold_n=.05, minimum_opposition_cosine=-.25,
    minimum_absolute_face_projection=.2, torque_limit_fraction=.98,
    torque_threshold_role="diagnostic_only_not_native_success_criterion",
    torque_source=TORQUE_SOURCE, slip_threshold_m=.0005,
    normal_source="reduced_native_force_direction",
    style_masks={style: list(mask) for style, mask in STYLE_MASKS.items()},
    pad_event_digit_scope="all_ten_pads", reason_digit_scope="grasp_style_mask",
    valid_s_source="all style-selected pad load measurements; opposition, torque and slip have independent coverage",
    historical_all_five_definition_unchanged=True,
    historical_all_five_comparison="style masks restrict bearing digits; historical surface_targets uses all five",
    duration_definition="sum dt_s per live measured env/substep; not wall time or cross-episode elapsed time",
    unknown_definition="invalid sample censors continuity; unknown is neither contact loss nor zero slip",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, name, *, minimum=None, positive=False):
    _require(type(value) in (float, int) and math.isfinite(value), f"finite {name} required")
    _require(minimum is None or value >= minimum, f"{name} below minimum")
    _require(not positive or value > 0, f"positive {name} required")


def _integer(value, name):
    _require(type(value) is int and value >= 0, f"nonnegative integer {name} required")


def _boolean(value, name):
    _require(type(value) is bool, f"boolean {name} required")


def _vector(value, name, length=3, *, optional=True, boolean=False, positive=False):
    if value is None and optional:
        return
    _require(isinstance(value, list) and len(value) == length, f"{name} needs {length} entries")
    for component in value:
        if boolean:
            _boolean(component, name)
        else:
            _number(component, name, positive=positive)


def _json(value):
    """Reject NaN/Inf and non-JSON data even in uninterpreted raw evidence."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        _number(value, "JSON number")
        return
    if isinstance(value, list):
        for item in value:
            _json(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _require(type(key) is str, "JSON keys must be strings")
            _json(item)
        return
    raise ValueError("only JSON-compatible finite measurements accepted")


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _metric():
    return dict(valid_s=0., valid_samples=0, opposed_s=0., opposed_samples=0)


def _style_stats():
    return dict(live_s=0., valid_s=0., live_samples=0, valid_samples=0,
        per_hand={hand: _metric() for hand in HANDS}, bilateral=_metric(),
        torque=dict(valid_s=0., valid_samples=0, limited_s=0., limited_samples=0),
        slip=dict(applicable_s=0., applicable_samples=0, valid_s=0., valid_samples=0,
                  exceeded_s=0., exceeded_samples=0),
        reasons={reason: dict(flagged_s=0., flagged_samples=0, valid_s=0., valid_samples=0)
                 for reason in REASONS},
        pads={hand: {digit: dict(contact_s=0., established=0, persisted=0, lost=0, censored=0)
                     for digit in DIGITS} for hand in HANDS})


def _opposition(pads, contact_valid):
    """Pure Python equivalent of surface_targets(...)[paired] for a style mask.

    Choose the most opposed usable pair, then require every bearing digit to
    project unambiguously onto one of its two faces. Force balance is separate
    and is not asserted by this diagnostic.
    """
    bearing = [p for p in pads if p["accepted_load_n"] > .05 and p["valid"]]
    if not contact_valid:
        return None, False, len(bearing)
    units = []
    for pad in bearing:
        normal = pad["normal_on_hand_w"]
        length = math.sqrt(sum(v * v for v in normal)) if normal is not None else 0.
        if length <= 1e-12:
            return None, False, len(bearing)
        units.append([v / length for v in normal])
    if len(units) < 2:
        return False, True, len(bearing)
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    pairs = [(dot(a, b), i, j) for i, a in enumerate(units) for j, b in enumerate(units)]
    score, i, j = min(pairs)
    if score > -.25:
        return False, True, len(bearing)
    axis = [a - b for a, b in zip(units[i], units[j])]
    length = math.sqrt(dot(axis, axis))
    axis = [v / length for v in axis]
    projected = [dot(unit, axis) for unit in units]
    paired = (all(abs(v) > .2 for v in projected)
              and any(v > .2 for v in projected) and any(v < -.2 for v in projected))
    return paired, True, len(bearing)


class GraspContactDiagnostics:
    """Bounded reducer; sinks consume one env/substep record or one event.

    Only current identity metadata, ten pad states per environment and fixed
    group counters are retained. Sinks may persist raw evidence independently.
    ``observe`` validates an entire frame before changing state or streaming.
    Episode IDs must increase on reset; styles are fixed within an episode.
    """

    def __init__(self, run_id, *, sink=None, event_sink=None):
        _require(type(run_id) is str and bool(run_id), "nonempty run_id required")
        _require(sink is None or callable(sink), "sink must be callable")
        _require(event_sink is None or callable(event_sink), "event_sink must be callable")
        self.run_id = run_id
        self._sink = sink
        self._event_sink = event_sink
        self._last_frame = None
        self._envs = {}
        self._active = {}
        self._stats = {style: _style_stats() for style in STYLE_MASKS}
        self._episodes_seen = 0
        self._event_counts = {name: 0 for name in
                              ("CONTACT_ESTABLISHED", "CONTACT_PERSISTED", "CONTACT_LOSS", "CONTACT_CENSORED")}
        self._finish_reason = None

    def _validate(self, frame):
        _require(isinstance(frame, dict), "frame dictionary required")
        _json(frame)
        for key in ("physics_step", "control_step", "control_substep"):
            _integer(frame.get(key), key)
        _number(frame.get("sim_time_s"), "sim_time_s", minimum=0.)
        _number(frame.get("dt_s"), "dt_s", positive=True)
        if self._last_frame is not None:
            previous = self._last_frame
            _require(frame["physics_step"] > previous["physics_step"], "physics_step must strictly increase")
            _require(frame["sim_time_s"] > previous["sim_time_s"], "sim_time_s must strictly increase")
            _require((frame["control_step"], frame["control_substep"]) >
                     (previous["control_step"], previous["control_substep"]), "control identity must strictly increase")
        rows = frame.get("rows")
        _require(isinstance(rows, list) and bool(rows), "nonempty rows required")
        identities = set()
        for row in rows:
            _require(isinstance(row, dict), "row dictionary required")
            for key in ("env_id", "episode_id"):
                _integer(row.get(key), key)
            env = row["env_id"]
            _require(env not in identities, "duplicate env row")
            identities.add(env)
            _require(row.get("grasp_style") in STYLE_MASKS, "supported grasp_style required")
            for key in ("live", "done", "native_success", "measurement_valid", "slip_valid", "slip_applicable"):
                _boolean(row.get(key), key)
            _require(type(row.get("slip_reason")) is str, "slip_reason required")
            if row.get("slip_m") is not None:
                _number(row["slip_m"], "slip_m", minimum=0.)
            _require(not row["slip_valid"] or row.get("slip_m") is not None, "valid slip requires measured slip_m")
            pads = row.get("pads")
            _require(isinstance(pads, list) and len(pads) == 10, "ten pads required")
            keys, bodies = set(), {}
            for pad in pads:
                _require(isinstance(pad, dict), "pad dictionary required")
                key = (pad.get("hand"), pad.get("digit"))
                _require(key[0] in HANDS and key[1] in DIGITS and key not in keys, "invalid or duplicate pad identity")
                keys.add(key)
                _boolean(pad.get("valid"), "pad valid")
                body = pad.get("body_id")
                _require(type(body) is int and (body >= 0 or (body == -1 and not pad["valid"])),
                         "body_id must be known, or -1 only for an invalid pad")
                bodies[key] = body
                for name in ("raw_load_n", "load_n", "accepted_load_n"):
                    _number(pad.get(name), name, minimum=0.)
                tolerance = 1e-9 * max(1., pad["raw_load_n"], pad["load_n"], pad["accepted_load_n"])
                _require(pad["accepted_load_n"] <= pad["load_n"] + tolerance and
                         pad["load_n"] <= pad["raw_load_n"] + tolerance, "pad load gates must not increase load")
                for name in ("position_w_m", "normal_on_hand_w"):
                    _vector(pad.get(name), name)
                for name in ("contact_count", "wrong_normal_count"):
                    _integer(pad.get(name), name)
                ids = pad.get("raw_contact_ids")
                _require(isinstance(ids, list), "raw_contact_ids list required")
                for value in ids:
                    _integer(value, "raw_contact_id")
            _require(isinstance(row.get("raw_contacts"), list) and
                     all(isinstance(c, dict) for c in row["raw_contacts"]), "raw_contacts dictionaries required")
            _vector(row.get("estimated_torque_nm"), "estimated_torque_nm", 12)
            _vector(row.get("effort_limit_nm"), "effort_limit_nm", 12, positive=True)
            _vector(row.get("torque_valid"), "torque_valid", 12, boolean=True)
            _require(row.get("torque_source") is None or type(row["torque_source"]) is str, "torque_source string or None")
            if env in self._envs:
                old = self._envs[env]
                _require(row["episode_id"] >= old["episode_id"], "episode_id must not decrease or be reused")
                if row["episode_id"] == old["episode_id"]:
                    _require(row["grasp_style"] == old["style"], "style fixed within an episode")
                    _require(not old["done"] or not row["live"], "finished episode must reset before becoming live")
                _require(all(body < 0 or old["bodies"][key] < 0 or body == old["bodies"][key]
                             for key, body in bodies.items()), "body identity must remain fixed")
        if self._envs:
            _require(identities <= set(self._envs), "unknown environment cannot enter a running stream")
            _require(all(env in identities or old["done"] for env, old in self._envs.items()),
                     "live environment cannot be omitted")

    def _identity(self, frame, row):
        return dict(run_id=self.run_id, env_id=row["env_id"], episode_id=row["episode_id"],
            grasp_style=row["grasp_style"], **{key: frame[key] for key in
            ("physics_step", "control_step", "control_substep", "sim_time_s", "dt_s")})

    def _event(self, kind, identity, pad_key, body, state, *, reason=None):
        event = dict(schema_version=SCHEMA_VERSION, event_type=kind, **identity,
            hand=pad_key[0], digit=pad_key[1], body_id=body,
            start_physics_step=state["start_physics_step"],
            start_sim_time_s=state["start_sim_time_s"],
            last_contact_physics_step=state["last"]["physics_step"],
            last_contact_sim_time_s=state["last"]["sim_time_s"],
            sampled_duration_s=state["sampled_duration_s"], sampled_contact_count=state["samples"])
        if reason is not None:
            event["reason"] = reason
        self._event_counts[kind] += 1
        stat = self._stats[state["last"]["grasp_style"]]["pads"][pad_key[0]][pad_key[1]]
        stat[{"CONTACT_ESTABLISHED": "established", "CONTACT_PERSISTED": "persisted",
              "CONTACT_LOSS": "lost", "CONTACT_CENSORED": "censored"}[kind]] += 1
        if self._event_sink is not None:
            self._event_sink(deepcopy(event))
        return event

    def _censor_env(self, env, reason):
        events = []
        for key in list(self._active):
            if key[0] == env:
                state = self._active.pop(key)
                events.append(self._event("CONTACT_CENSORED", state["last"], key[1:],
                                          state["body_id"], state, reason=reason))
        return events

    def observe(self, frame):
        """Return and optionally stream records for one complete physics substep."""
        _require(self._finish_reason is None, "cannot observe after finish")
        self._validate(frame)
        records = []
        for row in frame["rows"]:
            env, episode, style = row["env_id"], row["episode_id"], row["grasp_style"]
            identity = self._identity(frame, row)
            old = self._envs.get(env)
            events = []
            if old is None or old["episode_id"] != episode:
                if old is not None:
                    events += self._censor_env(env, "episode_changed")
                self._episodes_seen += 1
            elif self._last_frame is not None and frame["physics_step"] != self._last_frame["physics_step"] + 1:
                events += self._censor_env(env, "sample_gap")
            bodies = {(p["hand"], p["digit"]): p["body_id"] for p in row["pads"]}
            if old is not None:
                bodies = {key: body if body >= 0 else old["bodies"][key] for key, body in bodies.items()}
            episode_done = row["done"] or (old is not None and old["episode_id"] == episode and old["done"])
            self._envs[env] = dict(episode_id=episode, style=style, done=episode_done, bodies=bodies)
            if not row["live"]:
                events += self._censor_env(env, "row_not_live")
            for pad in row["pads"]:
                key = (env, pad["hand"], pad["digit"])
                state = self._active.get(key)
                valid = row["live"] and row["measurement_valid"] and pad["valid"]
                if not valid:
                    if state is not None:
                        self._active.pop(key)
                        events.append(self._event("CONTACT_CENSORED", state["last"], key[1:],
                            state["body_id"], state, reason="measurement_unknown"))
                    continue
                contact = pad["accepted_load_n"] > .05
                if contact:
                    kind = "CONTACT_PERSISTED" if state is not None else "CONTACT_ESTABLISHED"
                    if state is None:
                        state = dict(body_id=pad["body_id"], start_physics_step=frame["physics_step"],
                                     start_sim_time_s=frame["sim_time_s"], sampled_duration_s=0., samples=0)
                        self._active[key] = state
                    state["sampled_duration_s"] += frame["dt_s"]
                    state["samples"] += 1
                    state["last"] = identity
                    self._stats[style]["pads"][pad["hand"]][pad["digit"]]["contact_s"] += frame["dt_s"]
                    events.append(self._event(kind, identity, key[1:], pad["body_id"], state))
                elif state is not None:
                    self._active.pop(key)
                    events.append(self._event("CONTACT_LOSS", identity, key[1:], pad["body_id"], state))
            if row["done"]:
                events += self._censor_env(env, "episode_end")
            record = {**deepcopy(row), **identity, "schema_version": SCHEMA_VERSION, "events": events}
            self._diagnose(record)
            self._accumulate(record)
            if self._sink is not None:
                self._sink(deepcopy(record))
            records.append(record)
        self._last_frame = {key: frame[key] for key in
                            ("physics_step", "control_step", "control_substep", "sim_time_s", "dt_s")}
        return records

    def _diagnose(self, record):
        known = record["live"] and record["measurement_valid"]
        selected = {hand: [p for p in record["pads"] if p["hand"] == hand and
                           p["digit"] in STYLE_MASKS[record["grasp_style"]]] for hand in HANDS}
        per_hand = {}
        for hand, pads in selected.items():
            contact_valid = known and all(p["valid"] for p in pads)
            opposed, opposition_valid, count = _opposition(pads, contact_valid)
            per_hand[hand] = dict(contact_valid=contact_valid, opposition_valid=opposition_valid,
                                  opposed=opposed, bearing_count=count if contact_valid else None)
        contact_valid = all(p["contact_valid"] for p in per_hand.values())
        opposition_valid = all(p["opposition_valid"] for p in per_hand.values())
        torque_valid = (record["live"] and record["torque_source"] == TORQUE_SOURCE
                        and record["estimated_torque_nm"] is not None
                        and record["effort_limit_nm"] is not None and record["torque_valid"] is not None
                        and all(record["torque_valid"]))
        slip_valid = (record["live"] and record["slip_valid"] and record["slip_applicable"]
                      and record["slip_m"] is not None)
        wrong_normal = known and any(p["valid"] and p["wrong_normal_count"] > 0
                                     for pads in selected.values() for p in pads)
        wrong_valid = wrong_normal or (known and all(p["valid"] for pads in selected.values() for p in pads))
        required_loss = any(e["event_type"] == "CONTACT_LOSS" and
                            e["digit"] in STYLE_MASKS[record["grasp_style"]] for e in record["events"])
        loss_valid = record["live"] and (contact_valid or required_loss)
        bilateral = all(p["opposed"] for p in per_hand.values()) if opposition_valid else None
        counts = [p["bearing_count"] for p in per_hand.values()] if contact_valid else []
        record["per_hand"] = per_hand
        record["bilateral_opposed"] = bilateral
        record["source_validity"] = dict(contact=contact_valid,
            opposition={hand: p["opposition_valid"] for hand, p in per_hand.items()} | {"bilateral": opposition_valid},
            torque=bool(torque_valid), slip=bool(slip_valid), wrong_normal=bool(wrong_valid), contact_loss=bool(loss_valid))
        record["reason_validity"] = dict(NO_CONTACT=contact_valid, ONE_SIDE_CONTACT=contact_valid,
            WRONG_NORMAL=bool(wrong_valid), CONTACT_LOSS=bool(loss_valid), NON_OPPOSED=opposition_valid,
            TORQUE_LIMITED=bool(torque_valid), SLIP=bool(slip_valid))
        record["reasons"] = dict(
            NO_CONTACT=contact_valid and sum(counts) == 0,
            ONE_SIDE_CONTACT=contact_valid and sum(c > 0 for c in counts) == 1,
            WRONG_NORMAL=bool(wrong_normal),
            CONTACT_LOSS=required_loss,
            NON_OPPOSED=opposition_valid and not bilateral,
            TORQUE_LIMITED=bool(torque_valid and any(abs(t) >= .98 * limit for t, limit in
                zip(record["estimated_torque_nm"], record["effort_limit_nm"]))),
            SLIP=bool(slip_valid and record["slip_m"] > .0005))
        for pad in record["pads"]:
            valid = known and pad["valid"]
            pad["effective_contact"] = pad["accepted_load_n"] > .05 if valid else None
            pad["normal_source"] = "reduced_native_force_direction"

    def _accumulate(self, record):
        if not record["live"]:
            return
        stat = self._stats[record["grasp_style"]]
        dt = record["dt_s"]
        stat["live_s"] += dt
        stat["live_samples"] += 1
        if record["source_validity"]["contact"]:
            stat["valid_s"] += dt
            stat["valid_samples"] += 1
        entries = [(stat["per_hand"][hand], result["opposition_valid"], result["opposed"])
                   for hand, result in record["per_hand"].items()]
        entries.append((stat["bilateral"], record["source_validity"]["opposition"]["bilateral"], record["bilateral_opposed"]))
        for metric, valid, opposed in entries:
            if valid:
                metric["valid_s"] += dt
                metric["valid_samples"] += 1
                if opposed:
                    metric["opposed_s"] += dt
                    metric["opposed_samples"] += 1
        for reason, flagged in record["reasons"].items():
            metric = stat["reasons"][reason]
            if record["reason_validity"][reason]:
                metric["valid_s"] += dt
                metric["valid_samples"] += 1
                if flagged:
                    metric["flagged_s"] += dt
                    metric["flagged_samples"] += 1
        if record["source_validity"]["torque"]:
            stat["torque"]["valid_s"] += dt
            stat["torque"]["valid_samples"] += 1
            if record["reasons"]["TORQUE_LIMITED"]:
                stat["torque"]["limited_s"] += dt
                stat["torque"]["limited_samples"] += 1
        if record["slip_applicable"]:
            stat["slip"]["applicable_s"] += dt
            stat["slip"]["applicable_samples"] += 1
        if record["source_validity"]["slip"]:
            stat["slip"]["valid_s"] += dt
            stat["slip"]["valid_samples"] += 1
            if record["reasons"]["SLIP"]:
                stat["slip"]["exceeded_s"] += dt
                stat["slip"]["exceeded_samples"] += 1

    def summary(self):
        """Return a JSON-safe snapshot, with None for every missing denominator."""
        result = dict(schema_version=SCHEMA_VERSION, run_id=self.run_id, contract=deepcopy(CONTRACT),
            by_style=deepcopy(self._stats), episodes_seen=self._episodes_seen,
            event_counts=dict(self._event_counts), active_pad_contacts=len(self._active),
            finish_reason=self._finish_reason)
        for stat in result["by_style"].values():
            stat["coverage"] = _ratio(stat["valid_s"], stat["live_s"])
            stat["sample_coverage"] = _ratio(stat["valid_samples"], stat["live_samples"])
            for metric in [*stat["per_hand"].values(), stat["bilateral"]]:
                metric["coverage"] = _ratio(metric["valid_s"], stat["live_s"])
                metric["opposed_time_ratio"] = _ratio(metric["opposed_s"], metric["valid_s"])
                metric["opposed_sample_ratio"] = _ratio(metric["opposed_samples"], metric["valid_samples"])
            stat["torque"]["coverage"] = _ratio(stat["torque"]["valid_s"], stat["live_s"])
            slip = stat["slip"]
            slip["coverage"] = _ratio(slip["valid_s"], slip["applicable_s"])
            slip["applicability_coverage"] = _ratio(slip["applicable_s"], stat["live_s"])
            slip["not_applicable_s"] = stat["live_s"] - slip["applicable_s"]
            slip["not_applicable_samples"] = stat["live_samples"] - slip["applicable_samples"]
            for metric in stat["reasons"].values():
                metric["coverage"] = _ratio(metric["valid_s"], stat["live_s"])
                metric["flagged_sample_ratio"] = _ratio(metric["flagged_samples"], metric["valid_samples"])
        return result

    def finish(self, reason="bounded_probe_end"):
        """Censor remaining contacts at their last sample; never synthesize loss."""
        _require(type(reason) is str and bool(reason), "nonempty finish reason required")
        if self._finish_reason is None:
            for env in self._envs:
                self._censor_env(env, reason)
            self._finish_reason = reason
        return self.summary()
