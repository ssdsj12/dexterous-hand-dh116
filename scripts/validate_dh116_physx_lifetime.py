"""CPU-only independent lifecycle receipt audit; no grasp/Teacher admission."""
import hashlib
import json
import math
from pathlib import Path
import re

try:
    from scripts.run_dh116_work_packages import debugger_status
    from scripts.probe_dh116_physx_lifetime import OWNED_NAMES, VERSION
except ModuleNotFoundError:
    from run_dh116_work_packages import debugger_status
    from probe_dh116_physx_lifetime import OWNED_NAMES, VERSION

PHASES=("resources_acquired","owned_hand_and_solver_references_dropped",
        "before_stop","after_stop","before_app_close","after_app_close")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def require(ok, reason):
    if not ok:
        raise ValueError(reason)

def capture_evidence(output, phase):
    output=Path(output); prefix="gdb_"+phase
    receipt=read(output/(prefix+".json"))
    require(receipt.get("phase")==phase and receipt.get("all_commands_succeeded") is True
            and receipt.get("errors")==[], "incomplete GDB command capture")
    suffixes=("maps","bt.txt","threads.txt","registers.txt","instructions.txt","libraries.txt","mappings.txt")
    expected={prefix+"."+suffix for suffix in suffixes}
    hashes=receipt.get("artifact_sha256",{})
    require(set(hashes)==expected, "GDB artifact set incomplete")
    text={}
    for name in expected:
        path=output/name
        require(sha(path)==hashes[name], "GDB artifact SHA changed: "+name)
        text[name.removeprefix(prefix+".")]=path.read_text()
        require(text[name.removeprefix(prefix+".")].strip(), "empty GDB artifact")
    require(re.search(r"(?m)^#0\s",text["bt.txt"]) and
            re.search(r"(?m)^Thread\s",text["threads.txt"]) and
            re.search(r"(?m)^#0\s",text["threads.txt"]), "GDB stacks lack actual frames")
    pc=re.search(r"(?m)^rip\s+(0x[0-9a-fA-F]+)",text["registers.txt"])
    require(pc is not None and "=>" in text["instructions.txt"],"GDB register/instruction capture missing")
    require(re.search(r"0x[0-9a-fA-F]+\s+0x[0-9a-fA-F]+",text["libraries.txt"]) and
            re.search(r"0x[0-9a-fA-F]+\s+0x[0-9a-fA-F]+",text["mappings.txt"]),
            "GDB library/mapping capture missing")
    address=int(pc[1],16); mapped=None
    for line in text["maps"].splitlines():
        match=re.match(r"([0-9a-f]+)-([0-9a-f]+)\s+(\S+)\s+([0-9a-f]+)\s+\S+\s+\S+\s*(.*)",line)
        if match and int(match[1],16)<=address<int(match[2],16):
            mapped={"pc":hex(address),"mapping":line,
                    "file_offset":hex(address-int(match[1],16)+int(match[4],16))}
    require(mapped is not None,"GDB PC not attributable to full process maps")
    return {"phase":phase,"receipt_sha256":sha(output/(prefix+".json")),
            "artifact_sha256":hashes,"program_counter":mapped}

def validate_artifacts(output, outer_exit, cleanup):
    output,outer_exit=Path(output),Path(outer_exit)
    require(cleanup in ("registry","explicit"),"unknown cleanup case")
    outer=read(outer_exit); log=outer_exit.parent/"native.log"
    require(outer.get("outer_exit_code")==0 and outer.get("native_lock_observed") is True,
            "outer exit or native lock failed")
    duration=outer.get("duration_seconds")
    require(type(duration) in (int,float) and math.isfinite(duration) and 0<=duration<=180,
            "180 second qualification budget exceeded or unknown")
    require(outer.get("log_sha256")==sha(log),"outer log SHA changed")
    status=debugger_status(log)
    require(status.get("debugger_status_verified") is True and
            status.get("debugger_inferior_exit_code")==0 and status.get("debugger_inferior_signal") is None,
            "independent inferior exit is not zero")
    require(all(outer.get(k)==v for k,v in status.items()),"outer/debugger status mismatch")
    require("DH116_GDB_FATAL_SIGNAL" not in log.read_text(errors="replace")
            and not list(output.glob("gdb_fatal*")),"fatal capture present")
    boundaries=[json.loads(line) for line in (output/"initialization_boundaries.jsonl").read_text().splitlines()]
    require(len(boundaries)==2,"exactly two observed reset boundaries required")
    for row,phase,step in zip(boundaries,("before_reset","after_reset"),(0,2)):
        require(row.get("phase")==phase and row.get("sdk_step")==step
                and math.isclose(row["sdk_time_s"],step/480,rel_tol=0,abs_tol=2e-7),
                "reset boundary clock mismatch")
    result=read(output/"result.json")
    expected=dict(version=VERSION,case=cleanup,physics_steps=2,measurement_steps=0,rod_steps=0,
        ppo_updates=0,play_simulations=False,lifetime_released_before_close=True,failure=None,
        process_exit_verified=False,grasp_qualified=False,teacher_admission=False)
    require(all(k in result and result[k]==v for k,v in expected.items()),"result scope/exit identity differs")
    clocks=result["clocks"]
    require([r.get("phase") for r in clocks]==["before_stop","after_stop","before_app_close","after_app_close"],
            "lifecycle clock phases differ")
    require(clocks[0].get("clock_available") is True,"pre-STOP clock unavailable")
    for row in clocks:
        require(row.get("clock_available") in (True,False),"unknown clock availability")
        if row["clock_available"]:
            require(row.get("sdk_step")==2 and math.isclose(row["sdk_time_s"],2/480,rel_tol=0,abs_tol=2e-7),
                    "clock advanced or reset after initialization")
        else:
            require("sdk_step" not in row and "sdk_time_s" not in row,"unavailable clock contains fabricated values")
    acquisition=read(output/"acquisition.json")
    require(all(result["acquisition"].get(k)==v for k,v in acquisition.items()),"acquisition disk/result differ")
    required=dict(root_sim_aliases_identical=True,provider_bound_to_runtime=True,
        rod_tick=0,physics_step=2,measured_solver_step_performed=False,provider_snapshot_performed=False,
        native_environments=1,render_products_created=0,teacher_admission=False,grasp_qualified=False)
    require(all(k in acquisition and acquisition[k]==v for k,v in required.items()),"resource/solver acquisition differs")
    require(math.isfinite(acquisition["solver_time_s"]) and
            acquisition["solver_time_s"]==acquisition["solver_time_origin_s"],"rod solver clock advanced")
    before=read(output/"before_app_close.json"); closed=read(output/"app_close_returned.json")
    require(before.get("case")==cleanup and before.get("play_simulations") is False
            and before.get("lifetime_released") is True and before.get("clocks")==clocks[:3],
            "pre-close frozen evidence differs")
    require(closed==dict(case=cleanup,app_close_returned=True,process_exit_verified=False),"actual close return missing")
    ownership=[json.loads(line) for line in (output/"ownership.jsonl").read_text().splitlines()]
    stop=[row for row in ownership if row.get("phase")=="SDK STOP"]
    require(len(stop)==1 and stop[0].get("manager_view_released") is True,"manager root view survived STOP")
    liveness={}
    for phase in PHASES:
        state=read(output/(phase+".refs.json"))["resources"]
        require(set(OWNED_NAMES)<=set(state),"resource liveness missing")
        require(re.search(r"(?m)^[0-9a-f]+-[0-9a-f]+\s", (output/(phase+".maps")).read_text()),
                "phase full process map missing")
        liveness[phase]={name:state[name].get("alive") for name in OWNED_NAMES}
    require(all(v is True for v in liveness["resources_acquired"].values()),"initial resource weakrefs unavailable")
    require(all(v is False for v in liveness["before_app_close"].values()),"native owner live/unknown before close")
    require(all(v is False for v in liveness["after_app_close"].values()),"native owner live/unknown after close")
    capture=capture_evidence(output,"finalize")
    return dict(lifecycle_qualified=True,case=cleanup,physics_steps=2,measurement_steps=0,rod_steps=0,
                ppo_updates=0,grasp_qualified=False,teacher_admission=False,liveness=liveness,
                normal_finalize=capture,outer_exit_sha256=sha(outer_exit))

def audit_outcome(output, outer_exit, cleanup):
    """Persist useful independent partial evidence even if this paired case fails."""
    output=Path(output)
    try:
        result=validate_artifacts(output,outer_exit,cleanup)
    except (OSError,ValueError,KeyError,TypeError,OverflowError) as error:
        result=dict(lifecycle_qualified=False,case=cleanup,reason=f"{type(error).__name__}: {error}",
                    grasp_qualified=False,teacher_admission=False)
    result["available_artifact_sha256"]={path.name:sha(path) for path in sorted(output.glob("*"))
        if path.is_file() and not path.is_symlink()}
    return result
