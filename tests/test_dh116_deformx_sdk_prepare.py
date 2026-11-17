from pathlib import Path

import pytest

from scripts import prepare_dh116_deformx_sdk as prepare


def test_accepts_archive_link_relative_to_member_parent():
    assert prepare.relative_symlink("targets/x86_64-linux/bin/nvcc", "../../../bin/nvcc") == Path(
        "targets/x86_64-linux/bin/nvcc"
    )


@pytest.mark.parametrize("name,target", [
    ("lib/x", "../../escape"),
    ("../escape", "x"),
    ("lib/x", "/tmp/outside"),
    ("lib/x", "inside\\windows"),
])
def test_rejects_outside_or_nonportable_links(name, target):
    with pytest.raises(ValueError, match="archive symlink"):
        prepare.relative_symlink(name, target)


def test_merge_keeps_identical_files_and_rejects_collision(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "header.h").write_text("expected")
    (target / "header.h").write_text("expected")
    prepare.merge_component(source, target)
    (source / "header.h").write_text("different")
    with pytest.raises(ValueError, match="conflicting SDK path"):
        prepare.merge_component(source, target)
    assert (target / "header.h").read_text() == "expected"
