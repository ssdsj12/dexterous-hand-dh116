"""Temporary CPU audit dependencies never read real archived evidence."""
import importlib
import importlib.util
import json
from pathlib import Path

import pytest


def api():
    assert importlib.util.find_spec("cpu_audit_fixture") is not None
    return importlib.import_module("cpu_audit_fixture")


def project(tmp_path):
    scripts=tmp_path/"scripts";scripts.mkdir()
    config=tmp_path/"configs"/"pair.json";config.parent.mkdir()
    config.write_text(json.dumps({"reset_seed":{"sha256":"a"*64},"cpu_fixture":True}))
    operation=scripts/"operation.py"
    operation.write_text("""from contextlib import contextmanager
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONFIG='configs/pair.json'
def load_pair(project=ROOT):
    raise AssertionError('real archived read_locked reached')
@contextmanager
def scenario_scope(project=ROOT):
    packet=load_pair(project)
    yield packet
""")
    return operation,config


def load_module(path):
    namespace={"__file__":str(path)}
    exec(compile(path.read_text(),str(path),"exec"),namespace)
    return namespace


def test_local_pair_identity_replaces_archive_loader_and_preserves_scope(tmp_path):
    operation,config=project(tmp_path)
    before=operation.read_text()
    api().install_local_pair_loaders(tmp_path,["scripts/operation.py"])
    assert operation.read_text().startswith(before)
    namespace=load_module(operation)
    with namespace["scenario_scope"]() as packet:
        assert packet==json.loads(config.read_text())
    reads=json.loads((tmp_path/"cpu_fixture_pair_reads.jsonl").read_text())
    assert reads["config"]=="configs/pair.json" and reads["cpu_fixture"] is True


def test_pair_loader_rejects_project_escape(tmp_path):
    operation,_=project(tmp_path)
    api().install_local_pair_loaders(tmp_path,["scripts/operation.py"])
    with pytest.raises(ValueError,match="project"):
        load_module(operation)["load_pair"](tmp_path.parent)


def test_pair_loader_rejects_changed_local_config_identity(tmp_path):
    operation,config=project(tmp_path)
    api().install_local_pair_loaders(tmp_path,["scripts/operation.py"])
    config.write_text('{}')
    with pytest.raises(ValueError,match="identity"):
        load_module(operation)["load_pair"]()


@pytest.mark.parametrize("reject",[False,True])
def test_audit_probe_records_intended_result_before_receipt(tmp_path,reject):
    scripts=tmp_path/"scripts";scripts.mkdir()
    path=scripts/"audit.py"
    path.write_text("def audit_outcome(output,outer,cleanup):\n    return {'cpu_fixture':True,'lifecycle_qualified':"+str(not reject)+"}\n")
    if reject:(tmp_path/"reject").touch()
    api().instrument_cpu_audit(tmp_path,"scripts/audit.py")
    output=tmp_path/"output";output.mkdir()
    result=load_module(path)["audit_outcome"](output,tmp_path/"outer.json","explicit")
    assert result["cpu_fixture_audit_invoked"] is True
    assert result["reason"]==("cpu_fixture_requested_rejection" if reject else "cpu_fixture_audit_accepted")
    event=json.loads((output/"cpu_fixture_audit_calls.jsonl").read_text())
    assert event["before_receipt"] is True and event["reject_requested"] is reject


def test_audit_probe_rejects_nonfixture_result(tmp_path):
    scripts=tmp_path/"scripts";scripts.mkdir()
    path=scripts/"audit.py";path.write_text("def audit_outcome(*args): return {'lifecycle_qualified':True}\n")
    api().instrument_cpu_audit(tmp_path,"scripts/audit.py")
    with pytest.raises(AssertionError,match="CPU fixture"):
        load_module(path)["audit_outcome"](tmp_path,tmp_path/"outer.json","explicit")


@pytest.mark.parametrize("git_kind",["file","directory"])
def test_writers_reject_git_projects_without_changing_modules(tmp_path,git_kind):
    operation,_=project(tmp_path)
    audit=tmp_path/"scripts"/"audit.py";audit.write_text("def audit_outcome(*args): return {}\n")
    before={path:path.read_bytes() for path in (operation,audit)}
    if git_kind=="file":(tmp_path/".git").write_text("gitdir: /unreachable/cpu-fixture\n")
    else:(tmp_path/".git").mkdir()
    module=api()
    with pytest.raises(ValueError,match="temporary CPU fixture project"):
        module.install_local_pair_loaders(tmp_path,["scripts/operation.py"])
    with pytest.raises(ValueError,match="temporary CPU fixture project"):
        module.instrument_cpu_audit(tmp_path,"scripts/audit.py")
    assert all(path.read_bytes()==payload for path,payload in before.items())


def test_writers_reject_helper_repository_even_without_git_marker(tmp_path,monkeypatch):
    operation,_=project(tmp_path)
    audit=tmp_path/"scripts"/"audit.py";audit.write_text("def audit_outcome(*args): return {}\n")
    before={path:path.read_bytes() for path in (operation,audit)}
    module=api();monkeypatch.setattr(module,"__file__",str(tmp_path/"tests"/"cpu_audit_fixture.py"))
    with pytest.raises(ValueError,match="temporary CPU fixture project"):
        module.install_local_pair_loaders(tmp_path,["scripts/operation.py"])
    with pytest.raises(ValueError,match="temporary CPU fixture project"):
        module.instrument_cpu_audit(tmp_path,"scripts/audit.py")
    assert all(path.read_bytes()==payload for path,payload in before.items())
