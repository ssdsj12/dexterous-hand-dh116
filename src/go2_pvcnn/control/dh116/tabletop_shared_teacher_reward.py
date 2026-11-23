"""Shared-owner measurements feeding the unchanged original TrainingReward."""
import math

from .tabletop_training_reward import SourceStamp, Measurement, RewardEndpoint, SourceFailure
from .tabletop_teacher_pose_goal import measured_active_angles, ERROR_FIELDS


def endpoint(grasp, clock, *, distance_m, health, guard, hands, finger_goal, scope):
    if (grasp not in ("pinch","tripod","power") or finger_goal.grasp != grasp
            or scope not in ("cpu_fixture","owner_source_unqualified")
            or type(health) is not int or type(distance_m) not in (float,int)
            or not math.isfinite(distance_m) or distance_m < 0
            or guard is not None and (type(guard) is not str or not guard)):
        raise SourceFailure("same-grasp actual geometry/health/guard measurements required")
    reward_scope="cpu_fixture" if scope=="cpu_fixture" else "unqualified_source"
    def stamp(source):
        return SourceStamp(clock["episode_uid"],clock["physics_step"],clock["solver_time_s"],
            reward_scope,source,env_id=clock["env_id"],reset_uid=str(clock["reset_id"]),rod_uid=clock["rod_uid"])
    def measured(value,source,unit="dimensionless"):
        return Measurement(value,unit,stamp(source))
    distance_source=("NativePowerPoseApproach.fixed_material_station" if grasp=="power"
                     else "PolicyOwned"+grasp.title()+"Approach.fixed_material41")
    fields=dict(distance=measured(float(distance_m),distance_source,"m"),
        numerical_health=measured(health==0,"RopeBatch.last_error.same_owner"),
        constraint_failure=measured(guard is not None,"original_joint_force_clearance_and_policy_source_geometry_guards"),
        original_guard_joint_force_clearance_geometry=measured(guard is not None,
            "original_joint_force_clearance_and_policy_source_geometry_guards"))
    errors=finger_goal.max_errors_rad(measured_active_angles(hands))
    for name,error in zip(ERROR_FIELDS,errors,strict=True):
        fields[name]=measured(error,"sdk_named_active_q","rad")
    return RewardEndpoint(stamp("policy_owned_"+grasp+"_endpoint_v1"),
                          tuple(str(i) for i in range(41)),fields)
