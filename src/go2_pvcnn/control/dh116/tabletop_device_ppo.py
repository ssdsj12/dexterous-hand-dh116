"""Opt-in device collector/update for the unchanged structured Teacher/PPO.

The frozen trainer has CPU transfers and CPU index factories inside methods,
so the boundary methods are overridden here. The loss, advantage normalization,
Adam configuration, clipping, KL guard, rollback and counters retain the
original order. Training tensors remain on the chosen device. PyTorch RNG
snapshots and Adam's non-capturable scalar step retain their original control
state representation; no observation or rollout is copied through CPU.
"""
from copy import deepcopy
import random
import uuid

import numpy as np
import torch

from .tabletop_deformable_privileged_ppo import DeformablePPOTrainer
from .tabletop_device_rollout import DeviceOnPolicyRollout, own_device_tree
from .tabletop_privileged_ppo import _finite, _index
from .tabletop_squashed_gaussian import squashed_log_prob, squashed_entropy, latent_gaussian_kl


VERSION = "dh116-device-deformable-ppo-v1"


def _statistics(value):
    """Materialize only the final returned telemetry, never a training batch."""
    if isinstance(value,torch.Tensor):
        if value.ndim == 0:
            return float(value)
        return [_statistics(item) for item in value]
    if isinstance(value,dict):
        return {key:_statistics(item) for key,item in value.items()}
    if isinstance(value,list):
        return [_statistics(item) for item in value]
    return value


class DeviceDeformablePPOTrainer(DeformablePPOTrainer):
    """Use the original model and public trainer API with device-owned samples.

    Constructor arguments are exactly DeformablePPOTrainer's. Inherited source
    hash/model validation, optimizer construction, noise projection and session
    capabilities remain in force. Exact KL evaluates the requested complete
    batch in one actor forward, independent of legacy visual chunk settings.
    """

    def _new_rollout(self):
        # Resolve aliases such as 'cuda'/'cpu:0' to the model's actual device.
        self.device = self.model.log_std.device
        self.rollout = DeviceOnPolicyRollout(self.rollout_steps,self.num_envs,
            session_id=self.session_id,policy_version=self.policy_version,
            capability=self._capability,device=self.device)

    def _actor_mean(self, actor_obs, count):
        return self.model.latent_mean(self._actor_input(actor_obs))

    @torch.no_grad()
    def collect_step(self, actor_obs, critic_obs):
        if self._pending is not None:
            raise RuntimeError("pending transition must be recorded before another sample")
        if self.rollout.full:
            raise RuntimeError("complete rollout must be updated before collection")
        for name,observations in (("actor",actor_obs),("critic",critic_obs)):
            common,nodes,_ = self.model._parts(observations)
            if len(common) != self.num_envs:
                raise ValueError(f"{name} observation environment count differs")
            if len(self.rollout) and self.rollout._rows[0][f"{name}_obs"]["rod_nodes"].shape != nodes.shape:
                raise ValueError(f"{name} rod node shape changed within this rollout")
        owned_actor = own_device_tree(actor_obs,self.device)
        owned_critic = own_device_tree(critic_obs,self.device)
        mu = self._actor_mean(owned_actor,self.num_envs)
        _finite("latent mean",mu)
        if mu.shape != (self.num_envs,24):
            raise ValueError("policy latent mean must be [N,24]")
        std = self.model.log_std.exp().expand_as(mu)
        z = mu+std*torch.randn_like(mu)
        actions,log_prob = z.tanh(),squashed_log_prob(z,mu,std)
        value = self.model.value(owned_critic)
        _finite("critic value",value)
        if value.shape != (self.num_envs,):
            raise ValueError("critic value must be [N]")
        token = uuid.uuid4().hex
        self._pending = dict(actor_obs=owned_actor,critic_obs=owned_critic,
            actions=own_device_tree(actions,self.device),latent_actions=own_device_tree(z,self.device),
            old_log_prob=own_device_tree(log_prob,self.device),old_mu=own_device_tree(mu,self.device),
            old_std=own_device_tree(std,self.device),values=own_device_tree(value,self.device),
            collection_token=token,session_id=self.session_id,policy_version=self.policy_version)
        return dict(actions=actions.clone(),latent_actions=z.clone(),log_prob=log_prob.clone(),
            latent_mean=mu.clone(),latent_std=std.clone(),value=value.clone(),
            collection_token=token,policy_version=self.policy_version)

    def record_step(self, reward, done, next_value, *, truncated=None, bootstrap_mask=None,
            executed_actions=None, action_owner="policy", collection_token=None):
        if self._pending is None:
            raise RuntimeError("no pending policy sample")
        if action_owner != "policy":
            raise ValueError("action owner must be the current policy")
        if collection_token is not None and collection_token != self._pending["collection_token"]:
            raise ValueError("collection token mismatch")
        for name,value in (("reward",reward),("next_value",next_value)):
            _finite(name,value)
            if value.shape != (self.num_envs,):
                raise ValueError(name+" must be [N]")
        def boolean(name,value):
            if not isinstance(value,torch.Tensor) or value.dtype != torch.bool or value.shape != (self.num_envs,):
                raise ValueError(name+" must be boolean [N]")
            return own_device_tree(value,self.device)
        dones = boolean("done",done)
        trunc = torch.zeros_like(dones) if truncated is None else boolean("truncated",truncated)
        if bool((trunc & ~dones).any()):
            raise ValueError("truncated rows must also be done")
        if bool(trunc.any()) and bootstrap_mask is None:
            raise ValueError("truncation requires explicit terminal-observation bootstrap mask")
        bootstrap = ~dones if bootstrap_mask is None else boolean("bootstrap_mask",bootstrap_mask)
        if bool((bootstrap & dones & ~trunc).any()):
            raise ValueError("actual terminal success/failure cannot bootstrap")
        if executed_actions is not None:
            _finite("issued actions",executed_actions)
            issued = own_device_tree(executed_actions,self.device)
            if not torch.equal(issued,self._pending["actions"]):
                raise ValueError("issued normalized actions differ from sampled policy actions")
        row = dict(self._pending,rewards=own_device_tree(reward,self.device),done=dones,
            next_value=own_device_tree(next_value,self.device),bootstrap_mask=bootstrap)
        self.rollout._append(row,self._capability)
        self._pending = None
        self.counters["environment_steps"] += self.num_envs

    @torch.no_grad()
    def _check_frozen_policy(self, batch):
        mu = self.model.latent_mean(self._actor_input(batch["actor_obs"]))
        std = self.model.log_std.exp().expand_as(mu)
        z = batch["latent_actions"]
        actual = squashed_log_prob(z,mu,std)
        errors = {}
        for name,current,old in (("mean",mu,batch["old_mu"]),("std",std,batch["old_std"]),
                ("log_probability",actual,batch["old_log_prob"])):
            errors[name] = (current-old).abs().max().double()
            if not torch.allclose(current,old,atol=2e-5,rtol=2e-5):
                label = "likelihood" if name == "log_probability" else name
                raise RuntimeError("on-policy frozen "+label+" changed before PPO update")
        value = self.model.value(batch["critic_obs"])
        errors["value"] = (value-batch["values"]).abs().max().double()
        if not torch.allclose(value,batch["values"],atol=2e-5,rtol=2e-5):
            raise RuntimeError("critic policy changed during on-policy rollout")
        if not torch.equal(z.tanh(),batch["actions"]):
            raise RuntimeError("saved action/latent sample mismatch")
        return {f"max_abs_{key}_error":value for key,value in errors.items()}

    @torch.no_grad()
    def _minibatch_exact_kl(self, batch, indices):
        if indices.device != self.device:
            raise ValueError("exact KL indices must remain on the policy device")
        mu = self.model.latent_mean(self._actor_input(_index(batch["actor_obs"],indices)))
        std = self.model.log_std.exp().expand_as(mu)
        # Original sum reduction in float32, then the original Python-double
        # division reproduced on device. No sampled-approximate or chunk KL.
        total = latent_gaussian_kl(batch["old_mu"][indices],batch["old_std"][indices],mu,std).clamp_min(0.).sum()
        return total.double()/len(indices)

    def _optimizer_step(self, batch):
        if not self.config.kl_backtrack_steps:
            self.optimizer.step()
            self.project_noise()
            return self.config.lr,[]
        policy = deepcopy(self.model.state_dict())
        optimizer = deepcopy(self.optimizer.state_dict())
        rng = dict(torch=torch.get_rng_state().clone(),python=random.getstate(),
            numpy=deepcopy(np.random.get_state()),
            cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
        def rollback():
            self.model.load_state_dict(policy)
            self.optimizer.load_state_dict(deepcopy(optimizer))
            torch.set_rng_state(rng["torch"])
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            if rng["cuda"] is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])
        all_rows = torch.arange(len(batch["values"]),device=self.device)
        attempts = []
        for attempt in range(self.config.kl_backtrack_steps+1):
            lr = self.config.lr*.1**attempt
            self.optimizer.param_groups[0]["lr"] = lr
            try:
                self.optimizer.step()
                self.project_noise()
                kl = self._minibatch_exact_kl(batch,all_rows)
            except Exception:
                rollback()
                raise
            attempts.append(dict(lr=lr,exact_full_rollout_kl=kl))
            if kl <= self.config.target_kl:
                self.optimizer.param_groups[0]["lr"] = self.config.lr
                return lr,attempts
            rollback()
        return None,attempts

    def update(self):
        if self._pending is not None:
            raise RuntimeError("cannot update with pending transition")
        if not self.rollout.full:
            raise RuntimeError("complete nonempty rollout required")
        batch = self.rollout.batch(resolve_actor=False)
        frozen_check = self._check_frozen_policy(batch)
        returns,advantage = self.rollout.returns_and_advantages(self.config)
        advantage = ((advantage-advantage.mean())/advantage.std(unbiased=False).clamp_min(1e-8)).detach()
        total = len(advantage)
        trace,epoch_trace = [],[]
        early_stopped = False
        accepted_updates,optimized_samples = 0,0
        aggregate = {name:torch.zeros((),dtype=torch.float64,device=self.device)
            for name in ("policy_loss","value_loss","entropy","kl")}
        for epoch in range(self.config.epochs):
            epoch_start = len(trace)
            minibatches = torch.randperm(total,device=self.device).tensor_split(min(total,self.config.minibatches))
            for minibatch,indices in enumerate(minibatches):
                self.optimizer.zero_grad(set_to_none=True)
                entropy_noise = torch.randn(len(indices),24,device=self.device,dtype=batch["old_mu"].dtype)
                actor_obs = self._actor_input(_index(batch["actor_obs"],indices))
                mu = self.model.latent_mean(actor_obs)
                std = self.model.log_std.exp().expand_as(mu)
                z = batch["latent_actions"][indices]
                log_prob = squashed_log_prob(z,mu,std)
                log_ratio = log_prob-batch["old_log_prob"][indices]
                ratio = log_ratio.exp()
                adv = advantage[indices]
                policy_loss = -torch.minimum(ratio*adv,
                    ratio.clamp(1-self.config.clip,1+self.config.clip)*adv).mean()
                value = self.model.value(batch["critic_obs"][indices])
                old_value,target = batch["values"][indices],returns[indices]
                clipped_value = old_value+(value-old_value).clamp(-self.config.clip,self.config.clip)
                value_loss = torch.maximum((value-target).square(),(clipped_value-target).square()).mean()
                entropy = squashed_entropy(mu,std,entropy_noise).mean()
                loss = policy_loss+self.config.value_coef*value_loss-self.config.entropy_coef*entropy
                _finite("PPO loss",loss)
                loss.backward()
                with torch.no_grad():
                    kl = latent_gaussian_kl(batch["old_mu"][indices],batch["old_std"][indices],mu,std).mean()
                    sampled_approx_kl = (torch.expm1(log_ratio)-log_ratio).mean().double()
                    ratio_clipped_fraction = ((ratio-1.).abs()>self.config.clip).float().mean().double()
                metrics = {name:item.detach().double() for name,item in
                    (("policy_loss",policy_loss),("value_loss",value_loss),("entropy",entropy),("kl",kl))}
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.config.max_grad_norm,error_if_nonfinite=True)
                accepted_lr,guard_attempts = self._optimizer_step(batch)
                target = self.config.target_kl
                post_kl = (None if target is None or accepted_lr is None else self._minibatch_exact_kl(batch,indices))
                trace.append(dict(epoch=epoch,minibatch=minibatch,samples=len(indices),
                    pre_step_exact_kl=metrics["kl"],post_step_exact_kl=post_kl,
                    sampled_approx_kl=sampled_approx_kl,entropy=metrics["entropy"],
                    probability_ratio_clip_fraction=ratio_clipped_fraction,
                    gradient_norm_before_clip=norm.double(),accepted_lr=accepted_lr,kl_guard_attempts=guard_attempts))
                if accepted_lr is None:
                    early_stopped = True
                    break
                accepted_updates += 1
                optimized_samples += len(indices)
                for name,item in metrics.items():
                    aggregate[name] += item
                if target is not None and not guard_attempts and post_kl > target:
                    early_stopped = True
                    break
            epoch_rows = trace[epoch_start:]
            epoch_samples = sum(row["samples"] for row in epoch_rows)
            epoch_trace.append(dict(policy_epoch_index=epoch,attempted_minibatches=len(epoch_rows),
                accepted_minibatches=sum(row["accepted_lr"] is not None for row in epoch_rows),
                exact_full_rollout_kl=self._minibatch_exact_kl(batch,torch.arange(total,device=self.device)),
                sample_weighted_pre_step_entropy=sum(row["entropy"]*row["samples"] for row in epoch_rows)/epoch_samples,
                early_stopped=early_stopped,
                action_groups={name:dict(indices=list(range(lo,hi)),
                    log_std=self.model.log_std.detach()[lo:hi].clone(),
                    latent_std=self.model.log_std.detach().exp()[lo:hi])
                    for name,lo,hi in (("fingers12",0,12),("translation6",12,18),("rotation6",18,24))}))
            if early_stopped:
                break
        if accepted_updates:
            self.policy_version += 1
            self.counters["updates"] += 1
        self._new_rollout()
        return _statistics(dict({key:value/accepted_updates if accepted_updates else None
            for key,value in aggregate.items()},samples=total,optimized_samples=optimized_samples,
            minibatch_updates=accepted_updates,policy_version=self.policy_version,target_kl=self.config.target_kl,
            early_stopped=early_stopped,frozen_policy_check=frozen_check,minibatch_trace=trace,epoch_trace=epoch_trace))
