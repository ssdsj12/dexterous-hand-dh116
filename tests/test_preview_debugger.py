import pytest

def test_first_fatal_dump_is_bounded_and_not_repeated():
    from go2_pvcnn.control.dh116.tabletop_preview_debugger import FatalDumpBudget
    calls=[];output=[]
    def execute(command):
        calls.append(command)
        return "x"*2048
    budget=FatalDumpBudget(execute,output.append,max_bytes=1024)
    budget.record("SIGSEGV")
    first=len("".join(output).encode())
    budget.record("SIGSEGV")
    budget.record("SIGSEGV")
    assert calls
    assert budget.payload_bytes<=1024
    assert len("".join(output).encode())-first<300
    assert "TRUNCATED" in "".join(output)
    assert budget.repeated_fault is True

def test_cpu_abort_keeps_real_signal_status_without_isaac(tmp_path):
    import os,subprocess,sys,resource,json
    from go2_pvcnn.control.dh116.tabletop_preview_debugger import preview_debugger_wrapper
    metadata={};wrapper=preview_debugger_wrapper(tmp_path,metadata,interpreter=sys.executable)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="")
    def no_core():resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    # A script file, since -c probes intentionally bypass the debugger.
    child=tmp_path/"cpu_abort.py"
    child.write_text("import os,signal\n"
        "signal.signal(signal.SIGABRT,lambda *args:print('BAD_REENTRANT_CLOSE',flush=True))\n"
        "from go2_pvcnn.control.dh116.tabletop_preview_launcher import install_fatal_defaults\n"
        "install_fatal_defaults(signal)\n"
        "os.kill(os.getpid(),signal.SIGABRT)\n")
    result=subprocess.run([wrapper,str(child)],env=env,capture_output=True,text=True,timeout=30,preexec_fn=no_core)
    assert result.returncode!=0
    prefix="DH116_GDB_INFERIOR_STATUS "
    row=next(line[len(prefix):] for line in result.stdout.splitlines() if line.startswith(prefix))
    status=json.loads(row)
    assert status["signal"]==6 and status["exit_code"] is None
    assert result.stdout.count("DH116_GDB_FATAL_BACKTRACE SIGABRT")==1
    assert "BAD_REENTRANT_CLOSE" not in result.stdout
