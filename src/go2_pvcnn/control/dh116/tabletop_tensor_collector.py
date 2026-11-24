"""Teacher policy/PPO adapter over the shared native physical runtime."""
import json
from time import perf_counter

import torch

from .tabletop_device_ppo import DeviceDeformablePPOTrainer
from .tabletop_privileged_ppo import PPOConfig
from .tabletop_tensor_native_inputs import pack_shared_tensor_inputs
from .tabletop_tensor_native_teacher import TensorNativeTeacher
from .tabletop_tensor_physics_runtime import (NativeTensorPhysicsRuntime, RawTableClearance,
    SIDES, CLEARANCE_M, collision_clearance, mesh_reset, probe_copy, require_substep_guard)


class NativeTensorCollector(NativeTensorPhysicsRuntime):
    """Keep Teacher sampling/value/rollout separate from shared physical stepping."""

    def __init__(self, project, sim, batch, prepared, archive, seed, initial, bindings, runtime, profile):
        self._configure_physics(project, sim, batch, prepared, archive, seed, initial, bindings, runtime, profile)
        cfg = json.loads((project/f"configs/dh116_training/privileged_deformable_teacher_{self.style}_v1.json").read_text())
        self.model = TensorNativeTeacher(cfg, backend_profile=profile, num_envs=self.n,reward_config=self.config).to(self.device)
        pp = self.config.ppo
        self.trainer = DeviceDeformablePPOTrainer(self.model,PPOConfig(lr=pp["learning_rate"],gamma=pp["gamma"],
            lam=pp["gae_lambda"],clip=pp["clip"],epochs=pp["epochs"],minibatches=pp["minibatches"],
            value_coef=pp["value"],entropy_coef=pp["entropy"],max_grad_norm=pp["max_grad_norm"],
            target_kl=pp["target_kl"],kl_backtrack_steps=pp["kl_backtrack_steps"]),
            rollout_steps=32,num_envs=self.n,device=self.device)
        self._initialize_episode()

    def observation(self):
        stamp = self.source._boundary()
        cv,cm,ci = self.source.contexts()
        packet = pack_shared_tensor_inputs(self.state["q"],self.state["qd"],joint_names=self.joints,
            body_link_pose_w=self.state["base_link_pose_w"],body_com_velocity_w=self.state["base_link_com_velocity_w"],
            root_seed_w=self.initial["root_seed_w"],env_origins=self.origins,rod_positions_w=self.nodes,
            rod_velocities_w=self.velocity,rest_lengths_m=self.source.rest_lengths_m,task_context=cv,task_context_valid=cm,
            source_identity=stamp,context_identities=ci,backend_profile_sha256=stamp.profile_sha256,
            grasp=self.style,scope=self.source.scope)
        return packet.for_teacher(self.model,current_identity=self.source._boundary())


    def window(self, end, *, probe_reset=False):
        try:
            self._assert_usable()
            return self._window_impl(end, probe_reset=probe_reset)
        except BaseException:
            self._physical_failed = True
            raise

    def _window_impl(self,end,*,probe_reset=False):
        self.timings={}
        phase_start=perf_counter()
        if self.controls==0:
            self.first_sdk = self.runtime.physics_step
        obs = self.observation()
        sample = self.trainer.collect_step(obs,obs)
        actions, token = sample["actions"],sample["collection_token"]
        if probe_reset and torch.equal(actions[0],actions[1]):
            raise ValueError("probe requires independent sampled 24D lane actions")
        self.timings["observation_policy_sample"]=perf_counter()-phase_start
        phase_start=perf_counter()
        tokens = (token,)*self.n
        policy = f"{self.trainer.session_id}:{self.trainer.policy_version}"
        self.begin_action_window(actions, token=token, policy_uids=(policy,)*self.n, end=end)
        self.timings["reward_window_begin"]=perf_counter()-phase_start
        self.advance_action_window(actions,end)
        phase_start=perf_counter()
        self.record_window_request()
        with torch.no_grad():
            value=self.model.value(self.observation())
        timeout = torch.zeros(self.n, device=self.device, dtype=torch.bool)
        if probe_reset:
            timeout[1] = ~self.reward._terminal[1]
        full = len(self.trainer.rollout)+1 == self.trainer.rollout_steps
        packet=self.complete_action_window(truncated=timeout,buffer_full=full)
        self.trainer.record_step(packet.reward.float(),packet.done,value,truncated=packet.truncated,
            bootstrap_mask=~packet.terminated,executed_actions=actions,collection_token=token)
        self.snapshot("window_"+str(self.controls),dict(schemas=tuple(self.reward.schemas),
            packet=packet,post_value=value,actions=actions))
        self.timings["value_and_reward_settlement"]=perf_counter()-phase_start
        phase_start=perf_counter()
        if self._reset_allowed:
            ids=self._reset_allowed
            self.reset(ids)
        self.timings["window_reset"]=perf_counter()-phase_start
        return float(packet.reward.mean())

