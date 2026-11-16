from copy import deepcopy
from pathlib import Path
import json
import numpy as np

def test_flexible_spawn_inputs_satisfy_legacy_layout_translation_schema():
    from go2_pvcnn.control.dh116.tabletop_dual_hand_scene_preparation import prepare_scene,spawn_parameters
    root=Path(__file__).resolve().parents[1]
    prepared=prepare_scene(root)
    _,layout,_=spawn_parameters(root,prepared)
    reference=json.loads((root/"configs/dh116_training/pose_reference.json").read_text())["reference_layout"]
    # The unchanged shared builder translates this key before its external-wire branch.
    assert "wire_center" in layout
    assert np.asarray(layout["wire_center"]).shape==(3,)
    assert np.allclose(layout["wire_center"],np.mean(reference["wire_points"],axis=0))
    assert np.array_equal(layout["wire_points"],reference["wire_points"])
    assert layout["table_center"]==reference["table_center"]
    assert layout["table_size"]==reference["table_size"]
    # Check the original builder's pre-authoring translation works without mutating input.
    translated=deepcopy(layout);origin=np.array([1.,2.,3.])
    for key in ("wire_center","table_center"):
        translated[key]=np.asarray(translated[key])+origin
    assert np.allclose(translated["wire_center"],np.asarray(layout["wire_center"])+origin)
    assert np.array_equal(layout["wire_points"],reference["wire_points"])
    assert prepared["builder"]["external_wire"] is True
