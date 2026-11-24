"""New flat-source identity around the unchanged shared Teacher network/protocol."""
from pathlib import Path
from .tabletop_runtime_execution import EagerRuntimePolicy, validate_runtime_execution
from .tabletop_shared_native_teacher import SharedNativeTeacher
from .tabletop_native_teacher_model import initialize_native_network,INPUT_KEYS
from .tabletop_tensor_backend_profile import TensorBackendProfile
from .tabletop_native_teacher_profile import digest
from .tabletop_training_reward import TrainingReward,SMOKE

class TensorNativeTeacher(EagerRuntimePolicy, SharedNativeTeacher):
    def __init__(self,config,*,backend_profile,num_envs,reward_config):
        validate_runtime_execution(config=config)
        if type(backend_profile) is not TensorBackendProfile:
            raise ValueError("explicit independently verified flat CUDA profile required")
        backend_profile.verify()
        reward=TrainingReward(reward_config).config
        model=config.get("model",{});grasp=config.get("grasp")
        expected=dict(common_size=371,action_size=24,hidden=[256,256],rod_feature_size=13,rod_hidden=64)
        if (type(num_envs)is not int or not 2<=num_envs<=512 or grasp not in ("pinch","tripod","power")
                or config.get("task_kind")!="deformable_rod" or config.get("use_cameras") is not False
                or set(model)!=set(expected)|{"initial_std"} or any(model[k]!=v for k,v in expected.items())
                or type(model["initial_std"]) not in (int,float) or not .001<=model["initial_std"]<=.3
                or reward.profile!=SMOKE or reward.finger_goal is None or reward.finger_goal.grasp!=grasp):
            raise ValueError("unchanged shared finger-goal model/reward configuration required")
        identity=dict(policy_version="dh116-tensor-native-context-teacher-v1",
            source_family="shared_batched_solver_v1",grasp=grasp,task_kind="deformable_rod",
            parent_num_envs=num_envs,action_size=24,common_size=371,measured_common_size=66,
            raw_common_unknown="NaN_false_305",context_size=50,context_mode=SMOKE,input_keys=list(INPUT_KEYS),
            backend_sha256=backend_profile.binary_sha256,backend_profile_sha256=backend_profile.digest,
            source_tree_sha256=backend_profile.source_tree_sha256,config_semantic_sha256=digest(config),
            reward_config_sha256=reward.config_sha256,rod_position_frame="world_minus_environment_origin",
            other_rod_vector_frames="world",physics_coordinate_frame="world",
            checkpoint_namespace="tensor_native_teacher_"+grasp+"_v1",native_ready=False,
            teacher_admission=False,grasp_qualified=False,backend_scope="flat_candidate_unqualified",
            architecture=dict(common=[66,256,256],rod=[13,64,64],context=[100,64,64],
                              fusion=448,actor=24,critic=1))
        from .tabletop_stereo_contract import file_sha256
        identity["dependency_source_sha256"]={name:file_sha256(Path(__file__).with_name(name)) for name in (
            "tabletop_tensor_native_teacher.py","tabletop_native_teacher_model.py","tabletop_device_ppo.py",
            "tabletop_tensor_native_inputs.py","tabletop_tensor_backend_profile.py")}
        identity["model_identity_sha256"]=digest(identity)
        initialize_native_network(self,identity,backend_profile,SMOKE,model["initial_std"])
