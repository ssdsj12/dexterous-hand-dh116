"""Versioned shared-solver Teacher; reuses the existing network/distribution/PPO."""
from copy import deepcopy
from pathlib import Path

from torch import nn

from .tabletop_native_teacher_model import NativeContextTeacher, initialize_native_network, INPUT_KEYS
from .tabletop_native_teacher_profile import digest
from .tabletop_shared_teacher_source import SharedBackendProfile, SharedTeacherSource, FAMILY
from .tabletop_training_reward import TrainingReward, SMOKE

VERSION = "dh116-shared-native-context-teacher-v1"


class SharedNativeTeacher(NativeContextTeacher):
    """New source identity and observation frame; no legacy checkpoint relabelling."""

    def __init__(self, config, *, backend_profile, num_envs, reward_config):
        if type(backend_profile) is not SharedBackendProfile:
            raise ValueError("actual source-locked shared profile required")
        backend_profile.verify()
        # Same file-backed original reward, no numerical or guard changes.
        reward_config = TrainingReward(reward_config).config
        model = config.get("model", {})
        expected = dict(common_size=371, action_size=24, hidden=[256,256],
                        rod_feature_size=13, rod_hidden=64)
        grasp = config.get("grasp")
        if (type(num_envs) is not int or not 2 <= num_envs <= 512
                or grasp not in ("pinch","tripod","power") or config.get("task_kind") != "deformable_rod"
                or config.get("use_cameras") is not False
                or set(model) != set(expected)|{"initial_std"}
                or any(model[k] != v for k,v in expected.items())
                or type(model["initial_std"]) not in (float,int) or not .001 <= model["initial_std"] <= .3
                or reward_config.profile != SMOKE or reward_config.finger_goal is None
                or reward_config.finger_goal.grasp != grasp):
            raise ValueError("explicit shared finger-goal Teacher configuration required")
        identity = dict(policy_version=VERSION, source_family=FAMILY, grasp=grasp,
            task_kind="deformable_rod", parent_num_envs=num_envs, action_size=24,
            common_size=371, measured_common_size=66, raw_common_unknown="NaN_false_305",
            context_size=50, context_mode=SMOKE, input_keys=list(INPUT_KEYS),
            backend_sha256=backend_profile.binary_sha256, backend_profile_sha256=backend_profile.digest,
            source_tree_sha256=backend_profile.source_tree_sha256,
            config_semantic_sha256=digest(config), reward_config_sha256=reward_config.config_sha256,
            rod_position_frame="world_minus_environment_origin",
            other_rod_vector_frames="world", physics_coordinate_frame="world",
            checkpoint_namespace="shared_native_teacher_"+grasp+"_v1",
            native_ready=False, teacher_admission=False, grasp_qualified=False,
            architecture=dict(common=[66,256,256],rod=[13,64,64],context=[100,64,64],
                              fusion=448,actor=24,critic=1))
        dependencies = ("tabletop_native_teacher_model.py","tabletop_shared_native_teacher.py",
                        "tabletop_shared_teacher_source.py","tabletop_deformable_privileged_ppo.py")
        from .tabletop_stereo_contract import file_sha256
        identity["dependency_source_sha256"] = {
            n:file_sha256(Path(__file__).with_name(n)) for n in dependencies}
        identity["model_identity_sha256"] = digest(identity)
        initialize_native_network(self, identity, backend_profile, SMOKE, model["initial_std"])

    def pack(self, *args, **kwargs):
        raise ValueError("use explicit shared parent/row source capture")

    def pack_native_inputs(self, *args, **kwargs):
        raise ValueError("old one-solver-per-environment source cannot be relabelled")

    def pack_cpu_fixture(self, *args, **kwargs):
        raise ValueError("use explicitly CPU-scoped shared source")

    def pack_shared(self, packet, source):
        if type(source) is not SharedTeacherSource or source.profile != self._backend_profile:
            raise ValueError("same actual shared source profile required")
        source.validate_fresh(packet)
        if (packet.grasp != self.model_identity["grasp"]
                or packet.source.parent_num_envs != self.model_identity["parent_num_envs"]):
            raise ValueError("shared checkpoint grasp/environment contract differs")
        self._parts(packet.inputs)
        return {k:v.clone() for k,v in packet.inputs.items()}

    def set_extra_state(self, state):
        if state != self._model_identity:
            raise ValueError("shared Teacher checkpoint identity differs")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.set_extra_state(state_dict.get("_extra_state"))
        return nn.Module.load_state_dict(self, state_dict, strict=strict, assign=assign)
