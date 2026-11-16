"""Preview-only bounded fatal evidence; never fabricates a successful exit."""
import hashlib
from pathlib import Path
import shlex
import sys

POLICY_VERSION="dh116-preview-first-fatal-bounded-v2"

class FatalDumpBudget:
    def __init__(self,execute,write,*,max_bytes=8*1024*1024):
        if type(max_bytes) is not int or max_bytes<1:raise ValueError("positive byte limit required")
        self.execute,self.write=execute,write
        self.max_bytes,self.payload_bytes,self.fatal_count=max_bytes,0,0
    @property
    def repeated_fault(self):return self.fatal_count>=2
    def record(self,signal_name):
        self.fatal_count+=1
        if self.fatal_count==2:
            self.write("DH116_GDB_REPEATED_FATAL_PAUSED "+signal_name+"\n")
            return
        if self.fatal_count>2:return
        self.write("DH116_GDB_FATAL_BACKTRACE "+signal_name+"\n")
        for command in ("bt 64","info registers","x/16i $pc-16","thread apply all bt 32","info sharedlibrary","info proc mappings","py-bt"):
            self.write("DH116_GDB_COMMAND "+command+"\n")
            try:text=self.execute(command)
            except Exception as error:text="COMMAND_ERROR "+str(error)+"\n"
            raw=text.encode("utf-8",errors="replace")
            remaining=self.max_bytes-self.payload_bytes
            admitted=raw[:remaining]
            self.payload_bytes+=len(admitted)
            self.write(admitted.decode("utf-8",errors="ignore"))
            if len(admitted)<len(raw) or self.payload_bytes>=self.max_bytes:
                self.write("\nDH116_GDB_DUMP_TRUNCATED max_bytes="+str(self.max_bytes)+"\n")
                break

def preview_debugger_wrapper(path,metadata,*,interpreter=None):
    path=Path(path)
    root=Path(__file__).resolve().parents[4]
    script=r"""set pagination off
set debuginfod enabled off
set auto-load off
set print thread-events off
handle SIGSEGV stop print pass
handle SIGBUS stop print pass
handle SIGABRT stop print pass
python
import gdb,json,os,signal,sys,time
sys.path.insert(0,ROOT_SOURCE)
from go2_pvcnn.control.dh116.tabletop_preview_debugger import FatalDumpBudget
dump=FatalDumpBudget(lambda command:gdb.execute(command,to_string=True),gdb.write)
def on_stop(event):
    if isinstance(event,gdb.SignalEvent) and event.stop_signal in ("SIGSEGV","SIGBUS","SIGABRT"):
        dump.record(event.stop_signal)
gdb.events.stop.connect(on_stop)
def status(name):
    value=gdb.parse_and_eval(name)
    return None if str(value)=="void" else int(value)
end
run
python
while gdb.selected_inferior().pid:
    if dump.repeated_fault:
        # Leave inferior stopped. Do not auto-kill or turn a fault into exit0.
        os.kill(os.getpid(),signal.SIGSTOP)
        while True:time.sleep(60)
    gdb.execute("continue")
gdb.write("DH116_GDB_INFERIOR_STATUS "+json.dumps({
    "exit_code":status("$_exitcode"),"signal":status("$_exitsignal")})+"\n")
end
""".replace("ROOT_SOURCE",repr(str(root/"src")))
    commands=path/"preview-debugger.gdb"
    with commands.open("x") as stream:stream.write(script)
    python=shlex.quote(interpreter or sys.executable)
    wrapper=path/"preview-debugger-python"
    body='#!/usr/bin/env bash\nif [[ "${1:-}" == -c ]]; then\n exec '+python+' "$@"\nfi\nexec /usr/bin/gdb -nx -q --batch --return-child-result -x '+shlex.quote(str(commands))+' --args '+python+' "$@"\n'
    with wrapper.open("x") as stream:stream.write(body)
    wrapper.chmod(0o700)
    metadata.update(debugger_policy=POLICY_VERSION,
        debugger_script_sha256=hashlib.sha256(commands.read_bytes()).hexdigest(),
        debugger_wrapper_sha256=hashlib.sha256(wrapper.read_bytes()).hexdigest())
    return str(wrapper)
