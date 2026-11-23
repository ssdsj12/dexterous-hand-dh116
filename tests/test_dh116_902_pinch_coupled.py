"""CPU regression for inequality-aware Pinch corrections."""

import os
from pathlib import Path

import numpy as np
import pytest

from scripts import fit_dh116_902_pinch_coupled as coupled


def test_minimax_pairs_thumb_and_wrist_to_keep_proximal_clearance():
    # Thumb closure alone improves the first gap but consumes the proximal reserve.
    # The wrist coordinate restores clearance while worsening the index gap.
    gaps = np.array([.4, .2])
    gap_derivatives = np.array([[-1., 0.], [0., .2]])
    reserves = np.array([.01])
    reserve_derivatives = np.array([[-1., 1.]])
    step, active = coupled.minimax_step(gaps, gap_derivatives, reserves, reserve_derivatives,
                                        np.eye(2), np.full(2, -.3), np.full(2, .3), .25)
    assert step[0] > .15 and step[1] > .1
    assert np.min(reserves + reserve_derivatives @ step) >= -1e-9
    assert np.max(gaps + gap_derivatives @ step) < .27
    assert active == [0]



def test_extra_direction_keeps_the_entire_archived_node_group():
    from scripts.screen_dh116_tripod_supported_reset import archived_tube

    nodes = np.zeros((9, 41, 3))
    nodes[:, :, 1] = np.arange(41) * .01
    nodes[3, 5, 0] = .004
    tube = archived_tube(nodes)
    _, upper = coupled.add_support_directions(tube, nodes, [[1., 0., 0.]])
    assert upper[2, -1] == pytest.approx(.009)
    assert upper[1, -1] == pytest.approx(.005)


def test_probe_reuses_the_real_warm_fit_and_original_hulls():
    receipt = os.environ.get("DH116_PINCH_ALIGNMENT_RECEIPT")
    warm = os.environ.get("DH116_PINCH_COUPLED_WARM")
    if not receipt or not warm:
        pytest.skip("sealed 902 Pinch warm refit is external")
    result = coupled.fit(Path(receipt), Path(warm), side="right", iterations=2)
    assert result["sdk_range"] == [424, 432]
    assert result["warm"]["pad_gaps_m"] == pytest.approx([.000431123251285, .000215250704027])
    assert result["best"]["pad_stations"] == [4, 4]
    assert len(result["best"]["rod_envelopes"]) == 22
    assert result["best"]["active_q_rad"][3:] == result["warm"]["active_q_rad"][3:]
    assert max(result["best"]["pad_gaps_m"]) < max(result["warm"]["pad_gaps_m"])
    assert all(label.split(":")[1] in ("finger14_link", "finger23_link")
               for label in result["additional_rod_witnesses"])
    assert result["best"]["rod_envelopes"]["source_colliders:finger13_link"]["lower_m"] > .00025
    assert result["command_path_qualified"] is False
