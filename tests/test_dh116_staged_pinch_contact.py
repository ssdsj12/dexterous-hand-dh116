import numpy as np
import pytest

from scripts.audit_dh116_pinch_staged_closure import compile_close
from scripts.run_dh116_staged_pinch_contact import same_substep_reaction, schedule
from go2_pvcnn.control.dh116.tabletop_dual_hand_rod_runtime import MESH_NAMES


def _row(*, split=False, same_direction=False):
    all_nodes = np.zeros((1,41,3))
    total = {name: np.zeros(6) for name in MESH_NAMES}
    pairs = []
    for base in (0,6):
        for j in (0,1):
            step = int(split and base == 6)
            mesh = base+j
            normal = [1.,0.,0.] if j == 0 or same_direction else [-1.,0.,0.]
            force = .2*np.asarray(normal)
            point = [-.005,0.,0.] if j == 0 else [.005,0.,0.]
            pairs.append(dict(substep=step,node_index=mesh,mesh_index=mesh,mesh_name=MESH_NAMES[mesh],
                contact_point_world_m=point,outward_normal_world=normal,force_on_rod_world_n=force,
                torque_on_mesh_about_origin_world_nm=np.zeros(3)))
            all_nodes[0,mesh] += force/12
            total[MESH_NAMES[mesh]][:3] -= force/12
    pair = dict(physics_step=7,rod_tick=60,scope="rod_node_to_sdf_mesh_substep_force",
                substep_dt_s=1/(480*12),pairs=pairs)
    return dict(physics_step=7,rod_tick=60,clock_valid=True,mesh_names=MESH_NAMES,
                pair_capture_status="validated_solver_samples",rod_mesh_contact_pairs=pair,
                distributed_contact=all_nodes,named_wrenches=total)


def test_opposition_requires_both_hands_on_one_solver_substep():
    assert same_substep_reaction(_row(split=True))["bilateral_loaded_opposed_substeps"] == []
    evidence = same_substep_reaction(_row())
    assert evidence["bilateral_loaded_opposed_substeps"] == [0]
    assert evidence["all_12_loaded_opposed"] is False


def test_parallel_loaded_normals_do_not_qualify_as_opposed():
    assert same_substep_reaction(_row(same_direction=True))["bilateral_loaded_opposed_substeps"] == []


def test_mismatched_solver_aggregate_is_rejected():
    row = _row()
    row["distributed_contact"][0,0,0] = .04
    with pytest.raises(ValueError,match="reconcile"):
        same_substep_reaction(row)


def test_frozen_plan_yields_exactly_one_quiet_and_eight_close_windows():
    from pathlib import Path
    import json
    source = Path(__file__).resolve().parents[1] / "configs/dh116_execution/pinch-staged-open-and-close-archived-v1.json"
    plan = json.loads(source.read_text())["hands"]
    actions = schedule(plan)
    close = compile_close(plan)
    assert actions.shape == (9,24)
    assert np.all(actions[0]==0)
    assert np.array_equal(actions[1:],np.repeat(close[None,:],8,axis=0))
