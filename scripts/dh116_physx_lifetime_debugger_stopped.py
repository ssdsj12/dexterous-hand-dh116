"""Bound-output GDB capture for independent lifetime cases; starts no debugger."""
from pathlib import Path

COMMANDS = r'''set pagination off
set debuginfod enabled off
set auto-load off
set print thread-events off
set breakpoint pending on
set non-stop off
set environment DH116_PHYSX_LIFETIME_GDB 1
handle SIGSEGV stop print pass
handle SIGBUS stop print pass
python
import gdb, hashlib, json, os
from pathlib import Path
output = None
fatal_count = 0
finalize_count = 0

def capture(phase, stopped_thread_count):
    pid = gdb.selected_inferior().pid
    errors, files = [], {}
    try:
        raw = (Path("/proc") / str(pid) / "maps").read_bytes()
        target = output / ("gdb_" + phase + ".maps")
        with target.open("xb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        files[target.name] = hashlib.sha256(raw).hexdigest()
    except OSError as error:
        errors.append("maps: " + str(error))
    for suffix, command in (("bt", "bt 32"), ("threads", "thread apply all bt 20"),
                            ("registers", "info registers"), ("instructions", "x/8i $pc"),
                            ("libraries", "info sharedlibrary"), ("mappings", "info proc mappings")):
        gdb.write("DH116_GDB_COMMAND " + command + "\n")
        try:
            text = gdb.execute(command, to_string=True)
            if not text.strip():
                raise RuntimeError("empty command output")
            gdb.write(text)
            raw = text.encode()
            target = output / ("gdb_" + phase + "." + suffix + ".txt")
            with target.open("xb") as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            files[target.name] = hashlib.sha256(raw).hexdigest()
        except (gdb.error, OSError, RuntimeError) as error:
            errors.append(command + ": " + str(error))
            gdb.write("DH116_GDB_COMMAND_ERROR " + str(error) + "\n")
    result = {"phase": phase, "inferior_pid": pid, "stopped_thread_count": stopped_thread_count,
              "all_commands_succeeded": not errors,
              "errors": errors, "artifact_sha256": files, "process_exit_verified": False}
    try:
        with (output / ("gdb_" + phase + ".json")).open("x") as stream:
            json.dump(result, stream, indent=2); stream.write("\n")
            stream.flush(); os.fsync(stream.fileno())
    except OSError as error:
        gdb.write("DH116_GDB_COMMAND_ERROR receipt: " + str(error) + "\n")
        return False
    return not errors

finalize = gdb.Breakpoint("Py_FinalizeEx", internal=True)
finalize.silent = True
pending = None

def observe_stop(event):
    global pending
    if isinstance(event, gdb.SignalEvent) and event.stop_signal in ("SIGSEGV", "SIGBUS"):
        pending = ("fatal", event.stop_signal)
    elif isinstance(event, gdb.BreakpointEvent) and finalize in event.breakpoints:
        pending = ("finalize", None)
    else:
        pending = ("unhandled", None)

def status(name):
    value = gdb.parse_and_eval(name)
    return None if str(value) == "void" else int(value)

gdb.events.stop.connect(observe_stop)
gdb.execute("run")
while gdb.selected_inferior().pid:
    if pending is None or pending[0] == "unhandled":
        raise gdb.GdbError("unexpected inferior stop in lifetime capture")
    threads = gdb.selected_inferior().threads()
    current = gdb.selected_thread()
    if (not threads or current is None or not current.is_stopped()
            or any(not thread.is_stopped() for thread in threads)):
        raise gdb.GdbError("inferior has running threads after all-stop event")
    kind, signal_name = pending
    pending = None
    if kind == "fatal":
        fatal_count += 1
        gdb.write("DH116_GDB_FATAL_SIGNAL " + signal_name + "\n")
        if capture("fatal_" + str(fatal_count), len(threads)):
            gdb.write("DH116_GDB_FATAL_BACKTRACE " + signal_name + "\n")
    else:
        finalize_count += 1
        phase = "finalize" if finalize_count == 1 else "finalize_" + str(finalize_count)
        if capture(phase, len(threads)):
            gdb.write("DH116_GDB_NORMAL_FINALIZE_BACKTRACE\n")
    gdb.execute("continue")
gdb.write("DH116_GDB_INFERIOR_STATUS " + json.dumps({"exit_code": status("$_exitcode"), "signal": status("$_exitsignal")}) + "\n")
end
'''


def render_gdb_commands(output):
    output = Path(output)
    if not output.is_absolute() or output.resolve() != output:
        raise ValueError("canonical absolute GDB output required")
    return COMMANDS.replace("output = None", f"output = Path({str(output)!r})", 1)
