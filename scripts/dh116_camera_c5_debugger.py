"""Versioned C5 GDB commands; imported without starting a debugger."""

GDB_COMMANDS = r'''set pagination off
set debuginfod enabled off
set auto-load off
set print thread-events off
set breakpoint pending on
set environment DH116_CAMERA_C5_GDB 1
handle SIGSEGV stop print pass
handle SIGBUS stop print pass
python
import gdb
import json
import os
from pathlib import Path
import shlex

fatal_count = 0

def capture(phase):
    args = shlex.split(gdb.parameter("args"))
    output = Path(args[args.index("--output") + 1])
    pid = gdb.selected_inferior().pid
    errors = []
    maps_bytes = 0
    try:
        maps = Path("/proc") / str(pid) / "maps"
        content = maps.read_bytes()
        with (output / ("gdb_" + phase + ".maps")).open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        maps_bytes = len(content)
    except OSError as error:
        errors.append("full_proc_maps: " + str(error))
    for command in ("bt 32", "thread apply all bt 20", "info sharedlibrary", "info proc mappings"):
        gdb.write("DH116_GDB_COMMAND " + command + "\n")
        try:
            gdb.execute(command)
        except gdb.error as error:
            errors.append(command + ": " + str(error))
            gdb.write("DH116_GDB_COMMAND_ERROR " + str(error) + "\n")
    lifecycle = {}
    if phase == "finalize":
        for name in ("before_app_close", "app_close_returned"):
            try:
                lifecycle[name] = json.loads((output / (name + ".json")).read_text())
            except (OSError, ValueError) as error:
                errors.append(name + ": " + str(error))
    result = {"phase": phase, "inferior_pid": pid, "full_proc_maps_bytes": maps_bytes,
              "all_commands_succeeded": not errors, "errors": errors,
              "full_proc_maps_present": maps_bytes > 0, **lifecycle,
              "process_exit_verified": False}
    try:
        with (output / ("gdb_" + phase + ".json")).open("x") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        gdb.write("DH116_GDB_COMMAND_ERROR receipt: " + str(error) + "\n")

class FinalizeBreakpoint(gdb.Breakpoint):
    def stop(self):
        gdb.write("DH116_GDB_NORMAL_FINALIZE_BACKTRACE\n")
        capture("finalize")
        return False

finalize_breakpoint = FinalizeBreakpoint("Py_FinalizeEx", internal=True)
finalize_breakpoint.silent = True

def trace_fatal(event):
    global fatal_count
    if isinstance(event, gdb.SignalEvent) and event.stop_signal in ("SIGSEGV", "SIGBUS"):
        fatal_count += 1
        gdb.write("DH116_GDB_FATAL_BACKTRACE " + event.stop_signal + "\n")
        capture("fatal_" + str(fatal_count))

def status_variable(name):
    value = gdb.parse_and_eval(name)
    return None if str(value) == "void" else int(value)

gdb.events.stop.connect(trace_fatal)
end
run
python
while gdb.selected_inferior().pid:
    gdb.execute("continue")
gdb.write("DH116_GDB_INFERIOR_STATUS " + json.dumps({
    "exit_code": status_variable("$_exitcode"),
    "signal": status_variable("$_exitsignal"),
}) + "\n")
end
'''
