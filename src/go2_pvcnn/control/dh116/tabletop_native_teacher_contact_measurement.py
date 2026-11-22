"""Conservative same-boundary Teacher contact fields from the complete17 capture.

This is simulated DeformX reaction, not hardware force or grasp qualification.
All raw pair and support evidence survives; no contact is inferred from pose.
"""
from copy import deepcopy

from .tabletop_deformable_r0_replay import _digest
from .tabletop_power_contact_evidence import (
    measure_supported_distal_contact, MIN_LINK_LOAD_N, SUBSTEPS, TABLE_LOAD_LIMIT_N,
)


def capture_supported_contact(rod, *, grasp):
    """Validate the original solver packet and reduce every required substep."""
    evidence = measure_supported_distal_contact(rod, grasp=grasp)
    values = {}
    wrong_side = bool(evidence["wrong_side_contacts"])
    for side in ("left", "right"):
        hand = evidence["per_hand"][side]
        rows = hand["per_substep"]
        if len(rows) != SUBSTEPS or [row["substep"] for row in rows] != list(range(SUBSTEPS)):
            raise ValueError("complete ordered solver contact substeps required")
        required = hand["required_pads"]
        if not required:
            raise ValueError("grasp-specific required pads missing")
        values["coverage_"+side] = (0. if wrong_side else sum(
            all(row["per_pad"][pad]["normal_load_n"] > MIN_LINK_LOAD_N for row in rows)
            for pad in required) / len(required))
        values["opposed_"+side] = not wrong_side and all(row["effective"] for row in rows)
    # This retains the original >0.3N total-hand test as well as per-pad load,
    # opposition and material side. Coverage/opposition alone is insufficient.
    values["bilateral_support"] = not wrong_side and evidence["bilateral"]
    loads = deepcopy(evidence["external_support_force_norm_n"])
    peak = {name: max(samples) for name, samples in loads.items()}
    off_table = peak["table"] <= TABLE_LOAD_LIMIT_N
    return dict(version="dh116-supported-contact-measurement-v1", grasp=grasp,
        source="validated_complete17_DeformX_solver_pairs", identity=deepcopy(evidence["identity"]),
        timestamp_s=rod["timestamp_s"], raw_rod_sha256=_digest(rod),
        **values, off_table=off_table, no_external_support=evidence["support_free"],
        support_load_n=peak, support_substep_load_n=loads, table_contact_failure=not off_table,
        evidence=deepcopy(evidence), native_qualification=False, teacher_admission=False,
        grasp_qualified=False, force_proxy_qualified=False)
