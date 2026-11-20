"""CPU checks for the original A and separately authorized B launch contract."""
from copy import deepcopy

import pytest

from test_dh116_work_packages import repo
from test_native_teacher_actual_launch import actual_fixture, launch_module
from test_native_teacher_actual_resume_origin import archive, PACKAGE, RECEIPT_SHA


def test_b_companion_accepts_only_the_exact_a_receipt_reference(repo, tmp_path):
    package, _, launch = actual_fixture(repo, tmp_path)
    launch.update(role="b_resume", checkpoint_origin=dict(producer_package="teacher.a", receipt_sha256="d"*64))
    launch["provider"]["entrypoint"] = "run_b"
    frozen = launch_module().validate_actual_launch(launch, repository=repo, package=package)
    assert frozen == launch
    with pytest.raises(ValueError):
        tampered = deepcopy(launch)
        tampered["checkpoint_origin"]["checkpoint_path"] = "/not-a-frozen-receipt"
        launch_module().validate_actual_launch(tampered, repository=repo, package=package)


@pytest.mark.parametrize("damage", [None, "receipt", "source", "flags", "same_package", "bool"])
def test_b_origin_static_binding_uses_same_contract_and_never_grants_qualification(archive, damage):
    repo, _, _, _, _, consumer, _ = archive
    origin = repo.read_actual_resume_origin(PACKAGE, RECEIPT_SHA, resume_spec=consumer)
    launch = {"role":"b_resume", "checkpoint_origin":dict(producer_package=PACKAGE, receipt_sha256=RECEIPT_SHA)}
    name = "teacher.resume"
    if damage == "receipt":
        origin["receipt_sha256"] = "f"*64
    elif damage == "source":
        origin["source_closure_sha256"] = "e"*64
    elif damage == "flags":
        origin["teacher_admission"] = True
    elif damage == "same_package":
        name = PACKAGE
    elif damage == "bool":
        origin["loaded_updates"] = True
    match = launch_module().validate_actual_checkpoint_origin
    if damage:
        with pytest.raises(ValueError):
            match(launch, origin, consumer, package_name=name)
    else:
        match(launch, origin, consumer, package_name=name)
