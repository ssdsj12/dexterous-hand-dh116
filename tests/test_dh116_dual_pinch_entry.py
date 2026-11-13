"""Whole diagnostic operation with the original builder and CPU SDK doubles."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
from test_dh116_dual_pinch_builder import prepared,geometry,setup
from test_dh116_dual_pinch_interval import install_sdk_execution

ROOT=Path(__file__).resolve().parents[1]
MODULE="scripts.prepare_dh116_dual_pinch_interval"

def test_entry_describe_and_native_rejection_do_not_import_sdk_or_create_output(tmp_path):
    code="""import importlib.abc,sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in ('pxr','isaaclab','isaacsim','omni','ropesim'):
            raise RuntimeError('forbidden SDK import:'+name)
sys.meta_path.insert(0,Guard())
from scripts.prepare_dh116_dual_pinch_interval import main
main()
"""
    env={**os.environ,"CUDA_VISIBLE_DEVICES":""}
    result=subprocess.run([sys.executable,"-c",code],cwd=ROOT,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)["native_execution_available"] is False
    result=subprocess.run([sys.executable,"-c",code,"--execute-native","--output",str(tmp_path/"none")],
                          cwd=ROOT,env=env,capture_output=True,text=True)
    assert result.returncode!=0 and not (tmp_path/"none").exists()

@pytest.mark.parametrize("fault",[None,"pairs"])
def test_actual_builder_journal_eight_ticks_and_lifecycle_failure_boundary(monkeypatch,tmp_path,prepared,geometry,fault):
    entry=importlib.import_module(MODULE)
    from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as builder
    from go2_pvcnn.control.dh116.tabletop_deformable_dual_evidence import inspect_evidence
    sim,trace,reset_owners,sync,record,rows,events,refs,meta,drop=setup(
        monkeypatch,tmp_path,prepared,geometry)
    original=builder.build_reset_owned
    def sdk_build(sim,owners,**kwargs):
        try:
            result=original(sim,owners,**kwargs)
            install_sdk_execution(sim,owners.resources["runtime"],owners.resources["hand_bindings"],events,fault=fault)
            return result
        finally:sim=owners=None
    monkeypatch.setattr(builder,"build_reset_owned",sdk_build)
    output=tmp_path/"entry";output.mkdir()
    callbacks=[]
    app=SimpleNamespace(update=lambda:callbacks.append("update"),close=lambda:callbacks.append("close"))
    def clear_sim(value):
        drop();value.step=None;callbacks.append("clear")
    def execute():
        return entry.run_existing_app(app,make_sim=lambda:sim,synchronize=sync,
            clear_sim=clear_sim,stop=lambda:callbacks.append("stop"),
            manager_is_none=lambda:True,freeze=lambda:False,output=output,
            project=ROOT,prepared=prepared,owner_resets=reset_owners,
            run_uuid="cpu-entry",episode_uid="cpu-entry:paired")
    if fault:
        with pytest.raises(RuntimeError):execute()
        result=inspect_evidence(output/"physical.jsonl")
        assert result["measured_physics_rows"]==0 and result["unsettled_control_window"]
        assert sim.current_time_step_index==3
    else:
        result=execute();audit=entry.inspect_interval(output/"physical.jsonl")
        assert result["measurement_steps"]==8 and result["physics_steps"]==10
        assert audit["measured_physics_rows"]==8 and audit["complete_control_windows"]==1
        assert audit["static_table_contact_na_rows"]==8 and audit["no_hand_table_contact_proven"] is False
        assert audit["teacher_admission"] is False and audit["native_qualification"] is False
        assert sim.current_time_step_index==10
    assert callbacks==["clear","stop","update","close"]
    assert (output/"app_close_returned.json").exists()
    assert not any(ref() is not None for name,ref in refs.items())
