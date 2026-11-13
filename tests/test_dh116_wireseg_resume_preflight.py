import copy

import pytest

from scripts import preflight_dh116_wireseg_resume as preflight


def source(head):
    cohort=[dict(file_name=f"sample/{i}.png",family="fixture",pair_sha256=f"{i+1:064x}") for i in range(8)]
    return dict(state=dict(contract=dict(mode=head,identities=cohort),global_step=128),
                sampler=dict(visits_completed=158 if head=="depth-only" else 128,
                             next_index=6 if head=="depth-only" else 0),
                supervised=[i not in (0,4) for i in range(8)] if head=="depth-only" else [True]*8)


def test_separate_head_previews_preserve_cursor_and_skip_labels():
    seg,depth=source("seg-only"),source("depth-only")
    before=copy.deepcopy((seg,depth))
    segmentation=preflight.preview_visits(seg,4)
    metric_depth=preflight.preview_visits(depth,4)
    assert [row["visit"] for row in segmentation]==[129,130,131,132]
    assert all(row["updated"] for row in segmentation)
    assert [row["visit"] for row in metric_depth]==[159,160,161,162,163]
    assert [row["update_if_executed"] for row in metric_depth]==[129,130,None,131,132]
    assert metric_depth[2]["file_name"]=="sample/0.png"
    assert (seg,depth)==before


@pytest.mark.parametrize("damage", ["swapped_head","cohort","supervision","cursor"])
def test_rejects_mismatched_head_or_data(damage):
    state=source("depth-only")
    cohort=copy.deepcopy(state["state"]["contract"]["identities"])
    supervised=list(state["supervised"])
    expected="depth-only"
    if damage=="swapped_head":
        expected="seg-only"
    elif damage=="cohort":
        cohort.reverse()
    elif damage=="supervision":
        supervised[0]=True
    else:
        state["sampler"]["next_index"]=0
    with pytest.raises(ValueError):
        preflight.validate_head_support(state,expected,cohort,supervised)


def test_preview_has_no_update_without_supervision_or_unbounded_request():
    state=source("depth-only")
    state["supervised"]=[False]*8
    with pytest.raises(ValueError,match="supervision"):
        preflight.preview_visits(state,1)
    with pytest.raises(ValueError,match="1-16"):
        preflight.preview_visits(source("seg-only"),17)


def test_readonly_inspection_never_constructs_optimizer_or_runs_inference(monkeypatch, tmp_path):
    state=source("seg-only")
    state.update(origin_import_sha256="a"*64,source_checkpoint_sha256="b"*64,evidence={},
                 rng=None,evidence_sha256="c"*64)
    state["state"].update(optimizer={"state":{0:{"step":128.}}},rng="torch_cpu_saved",cuda_rng="cuda_saved")
    monkeypatch.setattr(preflight,"read_checkpoint",lambda *a: ({}, "a"*64))
    monkeypatch.setattr(preflight,"validate_source",lambda *a: state)
    monkeypatch.setattr(preflight,"verified_cohort",lambda *a: (
        state["state"]["contract"]["identities"],state["supervised"]))
    monkeypatch.setattr(preflight,"digest_file",lambda *a: "a"*64)
    record=preflight.inspect_head(tmp_path/"seg.pt","a"*64,tmp_path,"seg-only",4)
    assert record["mode"]=="seg-only" and record["global_step"]==128
    assert record["real_training_updates"]==0 and record["optimizer_constructed"] is False
    assert record["inference_views"]==0 and record["training_budget_authorized"] is False
    assert record["strict_resume_eligible"] is False
    assert record["required_execution_device_family"]=="cuda"
    assert record["unknown_historical_state"]==["python_rng","numpy_rng","complete_historical_runtime_config"]
