"""Real Ubuntu GDB tests, with a live background thread through Python finalization."""
import json
import os
import subprocess
import sys

import pytest

from scripts.dh116_physx_lifetime_debugger_stopped import render_gdb_commands
from scripts.run_dh116_work_packages import debugger_status
from scripts.validate_dh116_physx_lifetime import capture_evidence


@pytest.mark.parametrize("fatal,exit_code", [(False, 0), (False, 7), (True, None)])
def test_capture_multithreaded_inferior_in_stopped_context(tmp_path, fatal, exit_code):
    output = tmp_path / "captures"
    output.mkdir()
    script = tmp_path / "fixture.py"
    script.write_text(
        "import os, signal, threading, time\n"
        "threading.Thread(target=time.sleep, args=(30,), daemon=True).start()\n"
        + ("os.kill(os.getpid(), signal.SIGSEGV)\n" if fatal else f"raise SystemExit({exit_code})\n")
    )
    command = tmp_path / "debugger.gdb"
    command.write_text(render_gdb_commands(output))
    result = subprocess.run(
        ["/usr/bin/gdb", "-nx", "-q", "--batch", "--return-child-result", "-x",
         str(command), "--args", sys.executable, str(script)],
        cwd=tmp_path, text=True, capture_output=True, timeout=25,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    log = tmp_path / "fixture.log"
    log.write_text(result.stdout + result.stderr)
    assert debugger_status(log) == dict(
        debugger_status_verified=True, debugger_backtrace_captured=fatal,
        debugger_inferior_exit_code=exit_code, debugger_inferior_signal=11 if fatal else None,
    )
    if not fatal:
        assert result.returncode == exit_code
    phase = "fatal_1" if fatal else "finalize"
    capture = json.loads((output / f"gdb_{phase}.json").read_text())
    assert capture["stopped_thread_count"] >= 2
    assert capture["all_commands_succeeded"] is True
    assert capture_evidence(output, phase)["program_counter"]["pc"].startswith("0x")
