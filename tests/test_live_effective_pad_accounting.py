"""Negative evidence must retain original pair accounting while avoiding unused load reduction."""
from copy import deepcopy
import numpy as np
import pytest
from types import SimpleNamespace
from test_live_effective_pad_source import recorded,source,synthetic_complete_fixture
from go2_pvcnn.control.dh116.tabletop_complete_pair_capture import read_complete_pair_capture

def test_tiny_negative_normal_load_is_not_a_known_contact_absence(recorded,source):
    x=synthetic_complete_fixture(recorded[1]);rod=x["rod"]
    packet=rod["rod_mesh_contact_pairs"]["complete_capture"]
    raw=deepcopy(packet["raw"])
    raw["attempted"]=1;raw["indices"]=np.array([[0,0,5]])
    raw["vectors"]=np.array([[0,0,0,0,0,1,0,0,-1e-6,0,0,0]],dtype=float)
    raw["node_forces_on_rod_world_n"][:]=0
    raw["node_forces_on_rod_world_n"][0,2]=-1e-6/12
    raw["mesh_wrenches_on_mesh_world"][:]=0
    raw["mesh_wrenches_on_mesh_world"][5,2]=1e-6/12
    rod["distributed_contact"]=raw["node_forces_on_rod_world_n"][None]
    rod["named_wrenches"]={n:raw["mesh_wrenches_on_mesh_world"][i] for i,n in enumerate(packet["mesh_names"])}
    rod["rod_mesh_contact_pairs"]=read_complete_pair_capture(SimpleNamespace(contact_pair_samples=lambda:raw),
        dt_s=packet["dt_s"],physics_step=packet["physics_step"],rod_tick=packet["rod_tick"],
        mesh_names=packet["mesh_names"])
    with pytest.raises(ValueError):source.capture(x)
