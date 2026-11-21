"""Read-only process/source lease verification before any native construction.

This module never acquires the production reservation, issues authorization,
constructs an App or labels a run native-ready. Only the current actual FLOCK
holder can obtain this borrowed, nonserializable verification handle.
"""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import stat
import time

from .tabletop_native_teacher_worker_contract import _checked_binding, _sha, worker_registered_native_package

_TOKEN = object()


def _process_starttime(pid):
    raw = Path(f"/proc/{pid}/stat").read_text()
    return int(raw.rsplit(")",1)[1].split()[19])


def _verify(config, *, repo, package, expected_semantic_sha256):
    from scripts import run_dh116_work_packages as scheduler
    from .tabletop_native_teacher_preflight import validate_config, validate_budget, authorization_config_digest
    if type(repo) is not scheduler.Repository or not _sha(expected_semantic_sha256):
        raise ValueError("actual scheduler repository and semantic SHA required")
    try:
        package_name,current=worker_registered_native_package(repo,package)
        validate_config(config)
        _, auth_raw = _checked_binding(config["authorization"],project=repo.root)
        _, resource_raw = _checked_binding(config["resource_lock"],project=repo.root)
        auth = json.loads(auth_raw)
        resource = json.loads(resource_raw)
        if not isinstance(auth,dict) or not isinstance(resource,dict):
            raise ValueError("authorization and reservation must be JSON objects, never bool/list")
        pid = os.getpid()
        if type(auth.get("owner_pid")) is not int or auth["owner_pid"] != pid or resource.get("pid") != pid:
            raise ValueError("verification must run in the actual worker holder process")
        lock = scheduler.NATIVE_LOCK
        if lock.is_symlink() or not stat.S_ISREG(lock.stat().st_mode):
            raise ValueError("actual regular singleton lock inode required")
        info = lock.stat()
        # Preserve the original C01 config/auth/resource correspondence and
        # actual /proc/locks FLOCK owner check, rather than replacing it with JSON.
        budget = validate_budget(config,project=repo.root)
        starttime = _process_starttime(pid)
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        if (any(type(value) is not int or value != starttime for value in
                (auth.get("owner_starttime_ticks"),resource.get("owner_starttime_ticks")))
                or any(value != boot_id for value in (auth.get("boot_id"),resource.get("boot_id")))
                or type(resource.get("lock_device")) is not int or resource["lock_device"] != info.st_dev
                or type(resource.get("lock_inode")) is not int or resource["lock_inode"] != info.st_ino):
            raise ValueError("worker starttime/boot/device/inode lease identity differs")
        recipe = repo.native_recipe(current)
        spec = recipe["spec"]
        identity = {k:recipe[k] for k in ("command_spec_sha256","code_config_sha256")}
        contract = spec.get("teacher_contract",{})
        if contract.get("style") != config["style"] or contract.get("profile") != config["profile"]:
            raise ValueError("frozen recipe style/profile differs from the actual worker grasp")
        if (spec.get("version") != 2 or contract.get("semantic_sha256") != expected_semantic_sha256
                or auth.get("semantic_sha256") != expected_semantic_sha256
                or auth.get("recipe_identity") != identity or resource.get("recipe_identity") != identity
                or auth.get("source_sha256") != contract.get("source_sha256")
                or type(auth.get("max_ppo_updates")) is not int or auth["max_ppo_updates"] != 1):
            raise ValueError("current exact v2 source/config/semantic/update lease required")
        limits = contract["limits"]
        derived = dict(max_sdk_steps=spec["budget"]["physics_steps"],max_resets=limits["max_resets"],
            max_solver_calls=limits["max_solver_calls"],max_episode_count=limits["max_episode_count"],
            wall_timeout_s=limits["inner_wall_timeout_s"])
        if budget != derived:
            raise ValueError("C01 five-key budget must derive exactly from finite v2 recipe")
        _, frozen_raw = _checked_binding(contract["training_config"],project=repo.root)
        frozen = json.loads(frozen_raw)
        if (not isinstance(frozen,dict) or not isinstance(frozen.get("preflight"),dict)
                or frozen["preflight"].get("authorization") is not None
                or frozen["preflight"].get("resource_lock") is not None
                or authorization_config_digest(frozen["preflight"]) != authorization_config_digest(config)):
            raise ValueError("process config must match the pinned recipe template before lease overlay")
        issued = auth.get("issued_monotonic_s")
        deadline = auth.get("deadline_monotonic_s")
        now = time.monotonic()
        if (any(type(v) not in (int,float) or not math.isfinite(v) for v in (issued,deadline))
                or not 0 < issued <= now < deadline or deadline-issued > budget["wall_timeout_s"]):
            raise ValueError("worker monotonic wall lease expired or exceeds explicit recipe budget")
        return dict(package_name=package_name,pid=pid,starttime_ticks=starttime,boot_id=boot_id,lock_device=info.st_dev,
            lock_inode=info.st_ino,session_uid=auth["session_uid"],recipe_identity=identity,
            semantic_sha256=expected_semantic_sha256,source_sha256=contract["source_sha256"],
            authorization_sha256=config["authorization"]["sha256"],resource_sha256=config["resource_lock"]["sha256"],
            issued_monotonic_s=issued,deadline_monotonic_s=deadline)
    except (OSError,KeyError,TypeError,ValueError,IndexError) as error:
        raise ValueError("source-bound worker lease rejected: " + str(error)) from error


class WorkerLease:
    """Borrowed verification handle; closing it cannot release someone else's lock."""
    __slots__ = ("_config","_repo","_package","_semantic","_snapshot","_closed")

    def __init__(self, token=None, *, config=None, repo=None, package=None, semantic=None, snapshot=None):
        if token is not _TOKEN:
            raise TypeError("worker lease requires current process/kernel/source verification")
        self._config=deepcopy(config)
        self._repo=repo
        self._package=deepcopy(package)
        self._semantic=semantic
        self._snapshot=deepcopy(snapshot)
        self._closed=False

    @property
    def holder_pid(self):
        return self._snapshot["pid"]

    def revalidate(self):
        if self._closed:
            raise ValueError("worker lease is closed")
        observed = _verify(self._config,repo=self._repo,package=self._package,expected_semantic_sha256=self._semantic)
        if observed != self._snapshot:
            raise ValueError("worker lease identity changed after verification")
        return None

    def close(self):
        self._closed=True

    def __reduce__(self):
        raise TypeError("process-bound worker lease cannot be serialized")


def verify_worker_lease(config, *, repo, package, expected_semantic_sha256):
    """Verify an existing authorization/reservation; never create or acquire one."""
    snapshot = _verify(config,repo=repo,package=package,expected_semantic_sha256=expected_semantic_sha256)
    return WorkerLease(_TOKEN,config=config,repo=repo,package=package,semantic=expected_semantic_sha256,snapshot=snapshot)
