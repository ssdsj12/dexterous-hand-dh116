"""Versioned CPU diagnostics for Python ownership and archived native evidence.

No SDK import, release, collection, frame-local materialization or qualification.
Referrer output is bounded; gc.get_referrers itself enumerates direct Python
referrers and cannot reveal every C++/CUDA owner. A live weakref is not proof of
which reference caused the native crash.
"""
import gc
import hashlib
import json
from pathlib import Path
import re
import types
import weakref

VERSION = "dh116-native-lifetime-audit-v1"


def _type_name(value):
    kind = type(value)
    return kind.__module__ + "." + kind.__qualname__


def _describe_referrer(value, target):
    result = {"object_id": id(value), "type": _type_name(value)}
    if type(value) is dict:
        keys = []
        for index, (key, item) in enumerate(value.items()):
            if index >= 128:
                break
            if item is target and type(key) is str and len(keys) < 8:
                keys.append(key[:160])
        result.update(kind="dict", matching_keys=keys, keys_scan_truncated=len(value) > 128)
    elif type(value) is types.CellType:
        result["kind"] = "closure_cell"
    elif type(value) is types.FrameType:
        # Reading f_locals can itself extend native ownership. Never do it.
        result.update(kind="frame", function=value.f_code.co_name,
                      filename=value.f_code.co_filename, line=value.f_lineno)
    elif type(value) is types.FunctionType:
        result.update(kind="function", function=value.__qualname__,
                      filename=value.__code__.co_filename, line=value.__code__.co_firstlineno)
    elif type(value) is types.MethodType:
        result.update(kind="bound_method", function=value.__func__.__qualname__)
    else:
        result["kind"] = "other"
    return result


def snapshot_weakrefs(references, *, max_referrers=16):
    """Return scalar observations without retaining referents in the result.

    Call at an already authorized lifecycle boundary. This function does not
    collect or release anything; it must not be treated as a teardown fix.
    """
    if not 1 <= max_referrers <= 64 or not 1 <= len(references) <= 32:
        raise ValueError("diagnostic bounds require 1..32 owners and 1..64 referrers")
    if any(type(name) is not str or (ref is not None and type(ref) is not weakref.ReferenceType)
           for name, ref in references.items()):
        raise TypeError("only named weak references or explicit unsupported None are accepted")
    result = {"version": VERSION, "native_release_verified": False, "resources": {}}
    for name, ref in references.items():
        if ref is None:
            result["resources"][name] = {"alive": None, "reason": "weakref_unsupported"}
            continue
        target = ref()
        if target is None:
            result["resources"][name] = {"alive": False}
            continue
        referrers = gc.get_referrers(target)
        summaries = []
        # A comprehension capturing target would create a diagnostic-owned
        # closure cell and falsely report it as an application owner.
        for index in range(min(len(referrers), max_referrers)):
            summaries.append(_describe_referrer(referrers[index], target))
        result["resources"][name] = {
            "alive": True, "object_id": id(target), "type": _type_name(target),
            "referrer_count": len(referrers),
            "referrers_truncated": len(referrers) > max_referrers,
            "referrers": summaries,
        }
        del referrers, target
    return result


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_pinned_json(path, expected_sha):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("evidence SHA256 differs")
    return json.loads(raw)


def summarize_ownership(rows):
    phases = {row["phase"]: row for row in rows}
    before = phases.get("owned hand and solver references dropped", {}).get("python_liveness", {})
    after = phases.get("app.close returned, outer exit pending", {}).get("python_liveness", {})
    def alive(state):
        return sorted(name for name, value in state.items() if name != "simulation" and value is True)
    return {
        "manager_view_released": phases.get("SDK STOP", {}).get("manager_view_released"),
        "before_stop_liveness": before, "after_close_liveness": after,
        "live_before_stop": alive(before), "live_after_close": alive(after),
        "root_cause_proven": False,
        "limitations": "Python weakref liveness is separate from native release and causal attribution",
    }


def first_fatal_capture(log):
    """Read the first actual stack and mapping block; a marker alone is invalid."""
    marker = "DH116_GDB_FATAL_BACKTRACE "
    if marker not in log:
        return {"capture_complete": False, "reason": "no_fatal_marker", "frames": []}
    block = log.split(marker, 1)[1].split(marker, 1)[0]
    commands = ("bt 32", "thread apply all bt 20", "info sharedlibrary", "info proc mappings")
    present = all("DH116_GDB_COMMAND " + command + "\n" in block for command in commands)
    sections = {}
    positions = []
    for command in commands:
        token = "DH116_GDB_COMMAND " + command + "\n"
        if token in block:
            positions.append(block.index(token))
            sections[command] = block.split(token, 1)[1].split("DH116_GDB_COMMAND ", 1)[0]
    ordered = present and positions == sorted(positions)
    threads = sections.get("thread apply all bt 20", "")
    libraries = sections.get("info sharedlibrary", "")
    thread_content = bool(re.search(r"(?m)^Thread\s+\d+", threads)
                          and re.search(r"(?m)^#0\s+", threads))
    library_content = bool(re.search(r"(?m)^\s*0x[0-9a-fA-F]+\s+0x[0-9a-fA-F]+.* /", libraries))
    frames = []
    if "DH116_GDB_COMMAND bt 32\n" in block:
        stack = block.split("DH116_GDB_COMMAND bt 32\n", 1)[1].split("DH116_GDB_COMMAND ", 1)[0]
        for line in stack.splitlines():
            match = re.match(r"#(\d+)\s+(0x[0-9a-fA-F]+)\s+(.*)", line)
            if match:
                number, address, description = match.groups()
                frames.append(dict(number=int(number), pc=address, description=description,
                                   module=description.split(" from ", 1)[1] if " from " in description else None))
    maps = []
    if "DH116_GDB_COMMAND info proc mappings\n" in block:
        mapping = block.split("DH116_GDB_COMMAND info proc mappings\n", 1)[1]
        for line in mapping.splitlines():
            match = re.match(r"\s*(0x[0-9a-fA-F]+)\s+(0x[0-9a-fA-F]+)\s+"
                             r"(0x[0-9a-fA-F]+)\s+(0x[0-9a-fA-F]+)\s+(.*)", line)
            if match:
                start, end, size, offset, tail = match.groups()
                module = tail[tail.find("/"):] if "/" in tail else None
                maps.append(dict(start=int(start, 16), end=int(end, 16),
                                 offset=int(offset, 16), module=module))
    for frame in frames:
        pc = int(frame["pc"], 16)
        matches = [row for row in maps if row["start"] <= pc < row["end"] and row["module"] == frame["module"]]
        frame["file_offset"] = hex(pc - matches[0]["start"] + matches[0]["offset"]) if len(matches) == 1 else None
    complete = bool(ordered and thread_content and library_content and frames and frames[0]["number"] == 0 and frames[0]["file_offset"]
                    and "DH116_GDB_COMMAND_ERROR" not in block)
    return {
        "capture_complete": complete, "signal": block.splitlines()[0].strip(),
        "frames": frames, "mapping_rows": len(maps),
        "python_finalize_in_stack": any("Py_FinalizeEx" in row["description"] for row in frames),
        "registers_available": False,
    }
