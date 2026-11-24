"""Opt-in owned device rollout; original CPU trainer and source stay frozen.

The collector must retain its pending samples on this same device as well.
Installing this rollout alone does not remove the CPU copies in the original
DeformablePPOTrainer.collect_step/record_step/frozen-policy checks.
"""
import torch

from .tabletop_privileged_ppo import OnPolicyRollout, _tree, _join
from .tabletop_deformable_privileged_ppo import StructuredCriticBatch


VERSION = "dh116-device-on-policy-rollout-v1"
_NAMES = ("actor_obs", "critic_obs", "actions", "latent_actions", "old_log_prob",
    "old_mu", "old_std", "values", "rewards", "done", "next_value", "bootstrap_mask")


def own_device_tree(value, device):
    """Detach+clone on an already matching device; never stage a CPU copy."""
    device = torch.device(device)
    def own(tensor):
        if tensor.device != device:
            raise ValueError("rollout tensors must already be on the collection device")
        return tensor.detach().to(dtype=torch.float32 if tensor.is_floating_point() else tensor.dtype).clone()
    return _tree(value,own)


class DeviceOnPolicyRollout(OnPolicyRollout):
    """Same append/batch/GAE protocol; time-major flatten and terminal math.

    The append capability/session/policy gate is unchanged. Append additionally
    owns all tensor leaves, so a native reader reusing its frame cannot rewrite
    an earlier transition. ``batch`` also returns owned tensors. Structured
    critic batches preserve the original trainer's ``batch[indices].to`` API.
    """

    def __init__(self, rollout_steps, num_envs, *, session_id, policy_version,
            capability, actor_storage=None, device="cpu"):
        if any(type(v) is not int or v < 1 for v in (rollout_steps,num_envs)):
            raise ValueError("positive integer rollout_steps/num_envs required")
        self.device = torch.empty(0,device=device).device
        super().__init__(rollout_steps,num_envs,session_id=session_id,policy_version=policy_version,
            capability=capability,actor_storage=actor_storage)

    def _append(self, row, capability):
        if capability is not self._capability:
            raise ValueError("on-policy collection capability required")
        if self.full:
            raise RuntimeError("rollout is already complete")
        if row["session_id"] != self.session_id or row["policy_version"] != self.policy_version:
            raise ValueError("collection session/policy version mismatch")
        owned = dict(row)
        for name in _NAMES:
            def check(tensor):
                if tensor.ndim < 1 or tensor.shape[0] != self.num_envs:
                    raise ValueError("rollout tensor environment count differs: "+name)
                return tensor
            _tree(row[name],check)
            owned[name] = own_device_tree(row[name],self.device)
            if self._rows:
                previous = self._rows[0][name]
                current = owned[name]
                if isinstance(previous,dict) != isinstance(current,dict):
                    raise ValueError("rollout tensor schema changed: "+name)
                before = previous if isinstance(previous,dict) else {name:previous}
                after = current if isinstance(current,dict) else {name:current}
                if before.keys() != after.keys() or any(before[k].shape != after[k].shape or
                        before[k].dtype != after[k].dtype for k in before):
                    raise ValueError("rollout tensor shape/dtype schema changed: "+name)
        for name in ("done","bootstrap_mask"):
            if not isinstance(owned[name],torch.Tensor) or owned[name].shape != (self.num_envs,) or owned[name].dtype != torch.bool:
                raise ValueError(name+" must be boolean [N]")
        for name in ("rewards","values","next_value","old_log_prob"):
            if not isinstance(owned[name],torch.Tensor) or owned[name].shape != (self.num_envs,) or owned[name].dtype != torch.float32:
                raise ValueError(name+" must be float32 [N]")
        super()._append(owned,capability)

    def batch(self, *, resolve_actor=False):
        if not self._rows:
            raise RuntimeError("empty rollout")
        result = {name:_tree(_join([row[name] for row in self._rows]),lambda t:t.detach().clone()) for name in _NAMES}
        if resolve_actor and self.actor_storage is not None:
            result["actor_obs"] = own_device_tree(self.actor_storage.resolve(result["actor_obs"],self.device),self.device)
        if isinstance(result["critic_obs"],dict):
            result["critic_obs"] = StructuredCriticBatch(result["critic_obs"])
        return result

    def returns_and_advantages(self, config):
        if not self.full:
            raise RuntimeError("complete rollout required for GAE")
        advantages = []
        carry = torch.zeros(self.num_envs,device=self.device,dtype=torch.float32)
        for row in reversed(self._rows):
            delta = row["rewards"]+config.gamma*row["next_value"]*row["bootstrap_mask"]-row["values"]
            carry = delta+config.gamma*config.lam*(~row["done"]).float()*carry
            advantages.append(carry.clone())
        advantage = torch.stack(list(reversed(advantages))).flatten()
        values = torch.stack([row["values"] for row in self._rows]).flatten()
        return (advantage+values).detach(),advantage.detach()
