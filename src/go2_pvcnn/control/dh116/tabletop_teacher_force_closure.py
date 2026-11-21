"""Conservative force-closure evidence from complete DeformX solver pair captures.

This CPU reducer reports simulated contact reaction only. It makes no slip,
posture, gravity, dynamics, hardware, or native grasp-success inference.
"""
import itertools
import math

import numpy as np

from .tabletop_deformx_pair_capture import validate_pair_capture


_PAD_NAMES = ("thumb", "index", "middle", "ring", "little")
_REQUIRED = {"Pinch": 2, "Tripod": 3, "Power": 5}


def _pad_summary(pairs, divisor, mesh_id):
    load = sum(max(0., p["normal_force_on_rod_n"]) for p in pairs)
    weighted = np.zeros(3)
    force = np.zeros(3)
    for pair in pairs:
        weighted += max(0., pair["normal_force_on_rod_n"]) * np.asarray(pair["outward_normal_world"])
        force += pair["force_on_rod_world_n"]
    length = float(np.linalg.norm(weighted))
    normal = (weighted / length).tolist() if length > 1e-12 else None
    if not math.isfinite(load) or not np.isfinite(force).all() or not math.isfinite(length):
        raise ValueError("nonfinite reduced reaction")
    return dict(mesh_id=mesh_id, normal_load_n=float(load / divisor),
                force_on_rod_world_n=(force / divisor).tolist(),
                normal_on_rod_world=normal)


def _tripod_opposition(pads):
    names = ("thumb", "index", "middle")
    normals = [pads[name]["normal_on_rod_world"] for name in names]
    if any(normal is None for normal in normals):
        return dict(effective=False, opposition_cos=None, opposing_pads=[],
                    all_required_pads_on_opposing_faces=False)

    thumb, index, middle = np.asarray(normals)
    cosines = np.clip(np.array([thumb @ index, thumb @ middle]), -1., 1.)
    weakest = int(np.argmax(cosines))
    axis = thumb - (index + middle) / 2
    length = float(np.linalg.norm(axis))
    faces = False
    if length > 1e-12:
        axis /= length
        faces = bool(thumb @ axis >= .2 and index @ axis <= -.2 and middle @ axis <= -.2)
    return dict(effective=bool(faces and cosines[weakest] <= -.25),
                opposition_cos=float(cosines[weakest]), opposing_pads=["thumb", names[weakest + 1]],
                all_required_pads_on_opposing_faces=faces)


def _opposition(pads, grasp):
    if grasp == "Tripod":
        return _tripod_opposition(pads)
    candidates = []
    for first, second in itertools.combinations(pads, 2):
        a, b = pads[first]["normal_on_rod_world"], pads[second]["normal_on_rod_world"]
        if a is None or b is None:
            continue
        a, b = np.asarray(a), np.asarray(b)
        cosine = float(np.clip(np.dot(a, b), -1., 1.))
        candidates.append((cosine, first, second))
    eligible = [c for c in candidates if c[0] <= -.25]
    chosen = min(eligible or candidates, default=None)
    return dict(effective=bool(eligible),
                opposition_cos=None if chosen is None else chosen[0],
                opposing_pads=[] if chosen is None else list(chosen[1:3]),
                all_required_pads_on_opposing_faces=chosen is not None)


def reduce_force_closure(packet, *, grasp, dt_s=1/480):
    """Reduce the original atomic 13-mesh, one-rod solver capture.

    Grasp accepts lowercase runtime names and capitalized aliases.
    Mesh slots: left thumb..little 0..4, table 5, right thumb..little 6..10,
    supports 11/12. Each required pad must exceed .05 N, each hand must
    exceed .3 N, and weighted normals must oppose (cos <= -.25) in *every*
    solver substep. For Tripod, the thumb must oppose each partner and project
    by at least .2 onto the common axis, with both partners projecting by at
    least .2 in the opposite direction. Power needs actual palm or proximal
    contact, which this mesh layout does not measure.

    Per-pad loads and force vectors are averages over all solver substeps;
    per_substep contains the unaveraged evidence used for the decision.
    Invalid, incomplete, overflowing or unreconciled captures raise ValueError.
    """
    if isinstance(grasp, str) and grasp in ("pinch", "tripod", "power"):
        grasp = grasp.capitalize()
    if not isinstance(grasp, str) or grasp not in _REQUIRED:
        raise ValueError("grasp must be pinch, tripod or power (capitalized aliases accepted)")
    try:
        if packet["mesh_count"] != 13:
            raise ValueError("force closure requires the exact 13 mesh layout")
        # Do not filter/remap the packet or rebuild solver totals before validating.
        capture = validate_pair_capture(
            packet, dt_s=dt_s, node_forces=packet["node_forces_on_rod_world_n"],
            mesh_wrenches=packet["mesh_wrenches_on_mesh_world"])
    except (KeyError, TypeError, IndexError, OverflowError) as exc:
        raise ValueError("malformed complete solver pair packet") from exc
    return _summarize_distal_pairs(
        capture["pairs"], substeps=packet["substeps"], mesh_count=13, grasp=grasp,
        support_slots={"table": 5, "support_0": 11, "support_1": 12})


def _summarize_distal_pairs(pairs, *, substeps, mesh_count, grasp, support_slots):
    """Compute grasp-specific pad loading from previously reconciled pairs."""
    by_mesh = {mesh: [] for mesh in range(mesh_count)}
    for pair in pairs:
        by_mesh[pair["mesh_index"]].append(pair)
    hands = {}
    for side, base in (("left", 0), ("right", 6)):
        required = _PAD_NAMES[:_REQUIRED[grasp]]
        summaries = {name: _pad_summary(by_mesh[base + i], substeps, base + i)
                     for i, name in enumerate(_PAD_NAMES)}
        evidence, loaded, reasons = [], [], []
        for step in range(substeps):
            pads = {name: _pad_summary(
                [p for p in by_mesh[base + i] if p["substep"] == step], 1, base + i)
                for i, name in enumerate(required)}
            step_reasons = ["required_pad_normal_load_not_above_0.05n:" + name
                            for name, pad in pads.items() if pad["normal_load_n"] <= .05]
            total = sum(pad["normal_load_n"] for pad in pads.values())
            if total <= .3:
                step_reasons.append("required_hand_normal_load_not_above_0.3n")
            opposition = _opposition(pads, grasp)
            if not opposition["effective"]:
                step_reasons.append("required_pad_normal_opposition_missing")
            if grasp == "Power":
                step_reasons.append("power_palm_proximal_source_missing")
            passed = not step_reasons
            if passed:
                loaded.append(step)
            reasons.extend(step_reasons)
            evidence.append(dict(substep=step, per_pad=pads, total_normal_load_n=float(total),
                                 loaded=passed, reasons=step_reasons, **opposition))
        cosines = [step["opposition_cos"] for step in evidence]
        hands[side] = dict(
            loaded=len(loaded) == substeps, loaded_substeps=loaded,
            required_mesh_ids=[base + i for i in range(len(required))],
            required_pads=list(required), per_pad=summaries,
            total_normal_load_n=float(sum(summaries[name]["normal_load_n"] for name in required)),
            worst_opposition_cos=None if any(c is None for c in cosines) else max(cosines),
            per_substep=evidence, reasons=sorted(set(reasons)))
    support = {name: _pad_summary(by_mesh[mesh], substeps, mesh)
               for name, mesh in support_slots.items()}
    bilateral = all(hand["loaded"] for hand in hands.values())
    return dict(bilateral=bilateral, grasp=grasp, per_hand=hands,
                external_support=dict(per_mesh=support,
                    total_normal_load_n=float(sum(p["normal_load_n"] for p in support.values()))),
                source_scope="deformx_simulated_contact_reaction_not_hardware", native_success=False,
                solver_substeps=substeps, all_solver_substeps_loaded=bilateral)
