"""Same-state diagnostics for pose and reaction corrections in the wire Teacher."""
import torch

from .tabletop_deformable_r0_replay import _digest
from .tabletop_surface_contract import ACTION_CONTRACT, semantic_sha256
from .tabletop_wire_feedback_teacher import WireFeedbackTeacher


# Each canonical action block contains the left then the right hand.
_ACTION_GROUPS = {}
for name, key in (("fingers", "joint_velocity"),
                  ("translation", "world_translation_velocity"),
                  ("rotation", "intrinsic_rotation_velocity")):
    start, end = ACTION_CONTRACT["slices"][key]
    middle = (start + end) // 2
    _ACTION_GROUPS.update({f"left_{name}": slice(start, middle), f"right_{name}": slice(middle, end)})


@torch.inference_mode()
def compare_wire_feedback_actions(model, base_observation, rows, *, base_observation_sha256,
                                  clocks, pre_action_sha256s):
    """Compare one frozen policy on the same states with four feedback inputs.

    The zero-force counterfactual keeps the original validity bits and palm
    geometry. Its outputs measure numerical sensitivity on the logged states.
    They never become action labels, measured forces, or a physical outcome.
    """
    if type(model) is not WireFeedbackTeacher or any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("exact CPU wire-feedback Teacher required")
    if len(rows) == 0:
        raise ValueError("at least one source-bound observation required")
    weights = _digest(model.state_dict())
    measured = model.pack_recorded_feedback(
        base_observation, rows, base_observation_sha256=base_observation_sha256,
        clocks=clocks, pre_action_sha256s=pre_action_sha256s,
    )
    reaction = measured["wire_reaction"]
    reaction_valid = measured["wire_reaction_valid"]
    geometry_only = dict(measured, wire_reaction=torch.full_like(reaction, float("nan")),
                         wire_reaction_valid=torch.zeros_like(reaction_valid))
    zero_counterfactual = dict(measured, wire_reaction=torch.where(reaction_valid[..., None], 0., reaction))
    actions = {
        "feedback_absent": model.deterministic(model.missing_feedback(base_observation)),
        "geometry_only": model.deterministic(geometry_only),
        "geometry_with_zero_force_counterfactual": model.deterministic(zero_counterfactual),
        "measured_feedback": model.deterministic(measured),
    }
    for action in actions.values():
        if (action.shape != (len(rows), ACTION_CONTRACT["size"])
                or not bool(torch.isfinite(action).all() and (action.abs() <= 1.).all())):
            raise ValueError("expected finite bounded canonical 24D actions")
    if weights != _digest(model.state_dict()):
        raise ValueError("policy weights changed during the comparison")

    geometry = actions["geometry_only"]
    zero = actions["geometry_with_zero_force_counterfactual"]
    deltas = {
        "geometry": geometry - actions["feedback_absent"],
        "force_availability": zero - geometry,
        "force_value": actions["measured_feedback"] - zero,
    }
    group_max = {
        name: {group: value[:, indices].abs().amax(dim=1).tolist()
               for group, indices in _ACTION_GROUPS.items()}
        for name, value in deltas.items()
    }
    results = []
    for index, row in enumerate(rows):
        results.append(dict(
            clock=dict(clocks[index]),
            purpose=row["purpose"],
            status=row["status"],
            collection_token=row["collection_token"],
            pre_action_sha256=pre_action_sha256s[index],
            feedback_sha256=row["observation_sha256"],
            reaction_valid=reaction_valid[index].tolist(),
            action24={name: value[index].tolist() for name, value in actions.items()},
            correction24={name: value[index].tolist() for name, value in deltas.items()},
            absolute_max_by_group={name: {group: values[index] for group, values in groups.items()}
                                   for name, groups in group_max.items()},
        ))
    return dict(scope="same_state_cpu_policy_diagnostic", action_contract_sha256=semantic_sha256(ACTION_CONTRACT),
                grasp=model.model_identity["grasp"],
                policy_identity_sha256=model.model_identity["model_identity_sha256"],
                weights_sha256=weights, base_observation_sha256=base_observation_sha256,
                rows=results, native_runs=0, native_grasp_verified=False,
                teacher_admission=False, generalization_qualified=False)
