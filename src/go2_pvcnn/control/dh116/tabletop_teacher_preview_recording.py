"""Bounded CPU snapshots of native control frames and explicitly labelled replay.

No Isaac, torch, environment, scheduler, or physics callbacks are imported. Call
append once at the initial native state (policy_step=0), then once after each
actual control step. Clock values are observations supplied by the caller, never
invented here. A recorder belongs to one run/episode/environment/style/checkpoint.
This format attests byte integrity, not that a caller supplied genuine physics.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType
import numpy as np

IDENTITY_KEYS = ("run_uid", "episode_uid", "env_id", "style", "checkpoint_sha256")
ARRAY_KEYS = ("body_link_pose_w", "rod_nodes", "action", "net_lift")
SCHEMA = "dh116_native_preview_v1"


def _array(value):
    if value is None:
        raise ValueError("required measurement is missing")
    # Torch-like inputs are copied through CPU; no GPU storage is retained.
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    a = np.asarray(value)
    if a.dtype.kind not in "biuf" or not np.isfinite(a).all():
        raise ValueError("measurement must contain finite numeric values")
    # A bytes owner prevents callers re-enabling ndarray writes on cache storage.
    return np.frombuffer(a.tobytes(order="C"), dtype=a.dtype).reshape(a.shape)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(name + " must be a nonnegative integer")
    return int(value)


def _sha(meta, arrays):
    h = hashlib.sha256(json.dumps(meta, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    for name, a in sorted(arrays.items()):
        h.update(name.encode())
        h.update(a.dtype.str.encode())
        h.update(json.dumps(a.shape).encode())
        h.update(a.tobytes(order="C"))
    return h.hexdigest()


def _paths(path):
    p = Path(path)
    if p.suffix in (".json", ".npz"):
        p = p.with_suffix("")
    return Path(str(p) + ".npz"), Path(str(p) + ".json")


class NativePreviewRecorder:
    """Append-only, capacity-bounded collection; invalid appends are atomic."""

    def __init__(self, max_frames=481, control_hz=60):
        self.max_frames = _integer(max_frames, "max_frames")
        if not self.max_frames or not np.isfinite(control_hz) or control_hz <= 0:
            raise ValueError("positive capacity and control_hz required")
        self.control_hz = float(control_hz)
        self._frames = []
        self._sealed = False

    @property
    def frame_count(self):
        return len(self._frames)

    @property
    def identity(self):
        return self._frames[0]["identity"] if self._frames else None

    def append(self, *, run_uid, episode_uid, env_id, style, checkpoint_sha256,
               sdk_step, sdk_time, policy_step, body_link_pose_w, rod_nodes,
               action, net_lift, wrench=None, support_ledger=None, validity=None):
        if self._sealed:
            raise RuntimeError("recording already sealed")
        if self.frame_count >= self.max_frames:
            raise OverflowError("native preview capacity exhausted; frame rejected")
        env_id = _integer(env_id, "env_id")
        for value in (run_uid, episode_uid, style):
            if not isinstance(value, str) or not value:
                raise ValueError("nonempty run/episode/style identity required")
        if (not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64
                or any(c not in "0123456789abcdef" for c in checkpoint_sha256)):
            raise ValueError("checkpoint_sha256 must be a lowercase SHA256")
        identity = dict(zip(IDENTITY_KEYS, (run_uid, episode_uid, env_id, style, checkpoint_sha256)))
        sdk_step = _integer(sdk_step, "sdk_step")
        policy_step = _integer(policy_step, "policy_step")
        sdk_time = float(sdk_time)
        if not np.isfinite(sdk_time) or sdk_time < 0:
            raise ValueError("finite source SDK time required")
        if self._frames:
            last = self._frames[-1]
            if identity != last["identity"]:
                raise ValueError("cannot mix native identities or episodes")
            if sdk_step <= last["sdk_step"] or sdk_time <= last["sdk_time"]:
                raise ValueError("source SDK clock must strictly increase")
            if policy_step != last["policy_step"] + 1:
                raise ValueError("missing or repeated control frame; no interpolation permitted")
        elif policy_step != 0:
            raise ValueError("initial frame must have policy_step zero")
        arrays = {k: _array(v) for k, v in zip(ARRAY_KEYS,
                  (body_link_pose_w, rod_nodes, action, net_lift))}
        body, rod = arrays["body_link_pose_w"], arrays["rod_nodes"]
        if body.ndim != 3 or body.shape[0] != 2 or body.shape[1] < 1 or body.shape[2] != 7:
            raise ValueError("body_link_pose_w must have shape (2,B,7)")
        if rod.ndim != 2 or rod.shape[0] < 1 or rod.shape[1] != 3:
            raise ValueError("rod_nodes must have shape (K,3)")
        if arrays["action"].shape != (24,) or arrays["net_lift"].shape != ():
            raise ValueError("action must be actual 24D action; net_lift must be scalar")
        if self._frames:
            for k, a in arrays.items():
                if a.shape != self._frames[0][k].shape or a.dtype != self._frames[0][k].dtype:
                    raise ValueError("measurement shape/dtype changed within episode")
        actual_wrench = None if wrench is None else _array(wrench)
        if actual_wrench is not None and (actual_wrench.ndim != 2 or actual_wrench.shape[1] != 6):
            raise ValueError("wrench must have shape (M,6)")
        ledger = None
        if support_ledger is not None:
            if not isinstance(support_ledger, dict) or any(not isinstance(k, str) for k in support_ledger):
                raise ValueError("support_ledger must be a mapping of named numeric measurements")
            ledger = MappingProxyType({k: _array(v) for k, v in support_ledger.items()})
        if validity is not None and not isinstance(validity, (dict, type(MappingProxyType({})))):
            raise ValueError("validity must be a named boolean mapping")
        flags = {}
        for k, v in (validity or {}).items():
            if not isinstance(k, str) or not isinstance(v, (bool, np.bool_)):
                raise ValueError("validity must map names to actual boolean validity")
            flags[k] = bool(v)
        if flags.get("wrench", False) and actual_wrench is None:
            raise ValueError("missing wrench cannot be valid")
        if flags.get("support_ledger", False) and ledger is None:
            raise ValueError("missing support ledger cannot be valid")
        f = dict(arrays, identity=MappingProxyType(identity), sdk_step=sdk_step,
                 sdk_time=sdk_time, policy_step=policy_step, wrench=actual_wrench,
                 support_ledger=ledger, validity=MappingProxyType(flags))
        self._frames.append(MappingProxyType(f))

    def finish(self):
        if not self._frames:
            raise ValueError("cannot seal an empty native recording")
        self._sealed = True
        return NativePreviewRecording(tuple(self._frames), self.control_hz)


class NativePreviewRecording:
    """Read-only native observations. Save/load validates data and identity SHA."""

    def __init__(self, frames, control_hz):
        self._frames = frames
        self.control_hz = control_hz
        self._meta, self._arrays = self._pack()
        self.data_sha256 = _sha(self._meta, self._arrays)

    @property
    def identity(self):
        return self._frames[0]["identity"]

    @property
    def frame_count(self):
        return len(self._frames)

    def frame(self, index):
        if not isinstance(index, int) or not 0 <= index < self.frame_count:
            raise IndexError("recorded frame index out of range; cannot fabricate frames")
        return self._frames[index]

    def _stack(self, key):
        return _array(np.stack([f[key] for f in self._frames]))

    @property
    def body_link_pose_w(self):
        return self._stack("body_link_pose_w")

    @property
    def rod_nodes(self):
        return self._stack("rod_nodes")

    @property
    def action(self):
        return self._stack("action")

    @property
    def net_lift(self):
        return self._stack("net_lift")

    def _pack(self):
        arrays, frames = {}, []
        for i, f in enumerate(self._frames):
            d = {k: f[k] for k in ("sdk_step", "sdk_time", "policy_step")}
            d["validity"] = dict(f["validity"])
            d["arrays"] = {}
            for k in (*ARRAY_KEYS, "wrench"):
                if f[k] is None:
                    d["arrays"][k] = None
                else:
                    name = f"f{i}_{k}"
                    arrays[name] = f[k]
                    d["arrays"][k] = name
            d["support_ledger"] = None
            if f["support_ledger"] is not None:
                d["support_ledger"] = {}
                for j, (k, v) in enumerate(sorted(f["support_ledger"].items())):
                    name = f"f{i}_ledger_{j}"
                    arrays[name] = v
                    d["support_ledger"][k] = name
            frames.append(d)
        return dict(schema=SCHEMA, source="recorded_native_replay", control_hz=self.control_hz,
                    identity=dict(self.identity), frame_count=self.frame_count, frames=frames), arrays

    def save(self, path):
        npz_path, json_path = _paths(path)
        np.savez(npz_path, **self._arrays)
        metadata = dict(self._meta, data_sha256=self.data_sha256)
        json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n")
        return npz_path, json_path

    @classmethod
    def load(cls, path, *, max_frames=10000):
        npz_path, json_path = _paths(path)
        try:
            meta = json.loads(json_path.read_text())
            expected_sha = meta.pop("data_sha256")
            if meta["schema"] != SCHEMA or meta["source"] != "recorded_native_replay":
                raise ValueError("unsupported recording schema/source")
            n = _integer(meta["frame_count"], "frame_count")
            if not 0 < n <= max_frames or len(meta["frames"]) != n:
                raise ValueError("actual frame count mismatch or capacity exceeded")
            with np.load(npz_path, allow_pickle=False) as data:
                if len(data.files) != len(set(data.files)):
                    raise ValueError("duplicate NPZ arrays")
                arrays = {k: _array(data[k]) for k in data.files}
            if _sha(meta, arrays) != expected_sha:
                raise ValueError("recording data/identity SHA mismatch")
            r = NativePreviewRecorder(max_frames=n, control_hz=meta["control_hz"])
            used = set()
            for f in meta["frames"]:
                fields = {}
                for k in (*ARRAY_KEYS, "wrench"):
                    name = f["arrays"][k]
                    fields[k] = None if name is None else arrays[name]
                    if name is not None: used.add(name)
                ledger = f["support_ledger"]
                if ledger is not None:
                    fields["support_ledger"] = {k: arrays[v] for k, v in ledger.items()}
                    used.update(ledger.values())
                r.append(**meta["identity"], **fields, sdk_step=f["sdk_step"],
                         sdk_time=f["sdk_time"], policy_step=f["policy_step"], validity=f["validity"])
            rec = r.finish()
            if used != set(arrays) or rec.data_sha256 != expected_sha:
                raise ValueError("recording frame/schema/identity SHA mismatch")
            return rec
        except (KeyError, TypeError, OverflowError) as exc:
            raise ValueError("malformed native preview recording") from exc


class ReplayCursor:
    """Explicit UI events over CPU records only; no physics advancement API.

    Render current() after play/start before advance(). Each advance consumes
    exactly one recorded control frame, without interpolation or wall-clock catchup.
    play/start always restarts at frame zero, including after a completed replay.
    """

    def __init__(self, recordings):
        if not recordings:
            raise ValueError("at least one completed recording required")
        self._recordings = dict(recordings)
        for style, rec in self._recordings.items():
            if not isinstance(rec, NativePreviewRecording) or style != rec.identity["style"]:
                raise ValueError("style must match completed recording identity")
        self.style = next(iter(self._recordings))
        self.index = 0
        self.state = "waiting"
        self._events = []
        self._event("wait")

    @property
    def events(self):
        return tuple(MappingProxyType(dict(e)) for e in self._events)

    def _event(self, event):
        self._events.append(dict(event=event, style=self.style, frame_index=self.index,
                                 state=self.state, source="recorded_native_replay"))

    def select_style(self, style):
        if style not in self._recordings:
            raise KeyError(style)
        self.style, self.index, self.state = style, 0, "paused"
        self._event("select_style")
        return self.current()

    def start(self):
        self.index, self.state = 0, "playing"
        self._event("start")
        return self.current()

    def play(self):
        self.index, self.state = 0, "playing"
        self._event("play")
        return self.current()

    def wait(self):
        self.index, self.state = 0, "waiting"
        self._event("wait")
        return self.current()

    def pause(self):
        self.state = "paused"
        self._event("pause")

    def current(self):
        return self._recordings[self.style].frame(self.index)

    def advance(self):
        if self.state == "playing":
            if self.index + 1 < self._recordings[self.style].frame_count:
                self.index += 1
            else:
                self.state = "ended"
                self._event("ended")
        return self.current()

    def snapshot(self, *, frozen_sdk_step=None, frozen_sdk_time=None):
        f = self.current()
        return dict(source="recorded_native_replay", state=self.state, style=self.style,
                    frame_index=self.index, frame_count=self._recordings[self.style].frame_count,
                    identity=dict(f["identity"]), source_sdk_step=f["sdk_step"],
                    source_sdk_time=f["sdk_time"], source_policy_step=f["policy_step"],
                    frozen_sdk_step=frozen_sdk_step, frozen_sdk_time=frozen_sdk_time,
                    counts_as_native_evaluation=False, counts_as_native_success=False,
                    data_sha256=self._recordings[self.style].data_sha256)
