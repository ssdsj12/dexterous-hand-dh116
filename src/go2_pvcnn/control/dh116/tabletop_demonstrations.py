"""CPU storage of complete successful transitions from the real TensorEnv.

Call ``append`` with the observation saved *before* ``env.step(actions)``.
TensorEnv supplies the actual next observation in ``terminal_observation`` for
every row, including rows whose returned observation has already been reset.
Recording a success does not qualify it for imitation: repeat/physical proof
belongs in the separately supplied artifact qualification.
"""
from collections.abc import Mapping
from pathlib import Path

import torch


class SuccessEpisodeRecorder:
    def __init__(self, num_envs, observation_dim=156, action_dim=20, max_steps=1000):
        for name, value in (('num_envs', num_envs), ('observation_dim', observation_dim),
                            ('action_dim', action_dim), ('max_steps', max_steps)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        self.num_envs = num_envs
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.episodes = []
        self.episode_counts = torch.zeros(num_envs, dtype=torch.long)
        self._buffers = [[] for _ in range(num_envs)]
        self._overflow = [False] * num_envs
        self._overflow_episodes = 0
        self._control_steps = 0
        self._evidence_initialized = False
        self._evidence_schema = None

    @staticmethod
    def _tensor(value, name, shape, dtype=torch.float32):
        try:
            tensor = torch.as_tensor(value).detach().to(device='cpu')
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError(f'{name} must be a finite tensor with shape {shape}') from error
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError(f'{name} shape {tuple(tensor.shape)} != {tuple(shape)}')
        if tensor.is_complex() or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f'{name} must be finite and real')
        if dtype == torch.bool and not bool(((tensor == 0) | (tensor == 1)).all()):
            raise ValueError(f'{name} must contain exact boolean values')
        if dtype == torch.long and not bool(((tensor >= 0) & (tensor == tensor.floor())).all()):
            raise ValueError(f'{name} must contain nonnegative integer codes')
        tensor = tensor.to(dtype=dtype)
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f'{name} must remain finite in {dtype}')
        return tensor.clone()

    def _physical_evidence(self, value, rows):
        """Validate optional measured fields without inventing missing contacts.

        The collector represents missing contacts as finite zeros together with
        a false validity mask. Even masked NaNs are rejected here so sensor
        sanitation is explicit at the measured-evidence collection boundary.
        """
        if value is None:
            return None, None
        if not isinstance(value, Mapping) or not value:
            raise ValueError('physical_evidence must be a nonempty tensor mapping')
        evidence, schema = {}, {}
        for key, field in value.items():
            if not isinstance(key, str) or not key or not isinstance(field, torch.Tensor):
                raise ValueError('physical_evidence must have string keys and tensor fields')
            name = f'physical_evidence.{key}'
            if field.ndim < 1 or field.shape[0] != rows:
                raise ValueError(f'{name} shape must start with {rows}')
            dtype = torch.bool if field.dtype == torch.bool else torch.float32
            evidence[key] = self._tensor(field, name, field.shape, dtype)
            schema[key] = (tuple(field.shape[1:]), dtype)
        return evidence, schema

    def append(self, obs, actions, rewards, dones, extras):
        """Append one batched real transition; publish only complete successes.

        Validation happens before any counters or buffers change. Oversized
        episodes are discarded in full, and remain discarded until their done
        boundary so a tail can never masquerade as a complete trajectory.
        """
        if not isinstance(extras, Mapping):
            raise ValueError('extras must supply success and terminal_observation')
        for key in ('success', 'terminal_observation'):
            if key not in extras:
                raise ValueError(f'extras must contain {key}')
        n = self.num_envs
        batch = {
            'observations': self._tensor(obs, 'observations', (n, self.observation_dim)),
            'actions': self._tensor(actions, 'actions', (n, self.action_dim)),
            'next_observations': self._tensor(extras['terminal_observation'],
                'terminal_observation', (n, self.observation_dim)),
            'rewards': self._tensor(rewards, 'rewards', (n,)),
            'dones': self._tensor(dones, 'dones', (n,), torch.bool),
            'success_mask': self._tensor(extras['success'], 'success', (n,), torch.bool),
            'failure': self._tensor(extras.get('failure', torch.zeros(n, dtype=torch.long)),
                'failure', (n,), torch.long),
        }
        if 'phase' in extras:
            batch['phase'] = self._tensor(extras['phase'], 'phase', (n,), torch.long)
        evidence, schema = self._physical_evidence(extras.get('physical_evidence'), n)
        if self._evidence_initialized and schema != self._evidence_schema:
            raise ValueError('physical_evidence presence, keys, shapes and dtypes must remain consistent')
        self._evidence_initialized = True
        self._evidence_schema = schema
        self._control_steps += 1
        for env_index in range(n):
            buffer = self._buffers[env_index]
            if not self._overflow[env_index]:
                if len(buffer) == self.max_steps:
                    buffer.clear()
                    self._overflow[env_index] = True
                    self._overflow_episodes += 1
                else:
                    # Clone each row: retaining a view would keep every other
                    # environment's batch storage alive until this one resets.
                    step = {key: value[env_index].clone() for key, value in batch.items()}
                    if evidence is not None:
                        step['physical_evidence'] = {
                            key: value[env_index].clone() for key, value in evidence.items()}
                    buffer.append(step)
            if bool(batch['dones'][env_index]):
                success = (bool(batch['success_mask'][env_index])
                           and int(batch['failure'][env_index]) == 0)
                if (success and not self._overflow[env_index]
                        and all(int(step['failure']) == 0 for step in buffer)):
                    episode = {
                        'env_index': env_index,
                        'episode_index': int(self.episode_counts[env_index]),
                        'success': True,
                    }
                    for key in buffer[0]:
                        if all(key in step for step in buffer):
                            if key == 'physical_evidence':
                                episode[key] = {field: torch.stack([step[key][field] for step in buffer])
                                                for field in buffer[0][key]}
                            else:
                                episode[key] = torch.stack([step[key] for step in buffer])
                    self.episodes.append(episode)
                self.episode_counts[env_index] += 1
                buffer.clear()
                self._overflow[env_index] = False

    def summary(self):
        """Return JSON-compatible diagnostics, including failed reset counts."""
        completed = int(self.episode_counts.sum())
        return {
            'control_steps': self._control_steps,
            'completed_episodes': completed,
            'successful_episodes': len(self.episodes),
            'failed_episodes': completed - len(self.episodes),
            'overflow_episodes': self._overflow_episodes,
            'episode_counts': self.episode_counts.tolist(),
            'buffered_steps': [len(buffer) for buffer in self._buffers],
        }

    def _validated_episodes(self):
        if not self.episodes:
            raise ValueError('cannot save a dataset without successful episodes')
        episodes = []
        identities = set()
        for source in self.episodes:
            identity = (source.get('env_index'), source.get('episode_index'))
            if (any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in identity) or identity[0] >= self.num_envs or identity in identities):
                raise ValueError('successful episode identity is invalid or duplicated')
            identities.add(identity)
            if source.get('success') is not True:
                raise ValueError('dataset must contain only explicit successful episodes')
            observations = source.get('observations')
            if not isinstance(observations, torch.Tensor) or observations.ndim != 2:
                raise ValueError('observations shape must be (steps, observation_dim)')
            steps = observations.shape[0]
            if not 0 < steps <= self.max_steps:
                raise ValueError('successful episode shape exceeds step bounds')
            episode = {'env_index': identity[0], 'episode_index': identity[1], 'success': True}
            for key, shape, dtype in (
                    ('observations', (steps, self.observation_dim), torch.float32),
                    ('actions', (steps, self.action_dim), torch.float32),
                    ('next_observations', (steps, self.observation_dim), torch.float32),
                    ('rewards', (steps,), torch.float32),
                    ('dones', (steps,), torch.bool),
                    ('success_mask', (steps,), torch.bool),
                    ('failure', (steps,), torch.long)):
                episode[key] = self._tensor(source.get(key), key, shape, dtype)
            if (not bool(episode['dones'][-1]) or bool(episode['dones'][:-1].any())
                    or not bool(episode['success_mask'][-1]) or bool(episode['failure'].any())):
                raise ValueError('episode must end at an explicit successful, failure-free boundary')
            if 'phase' in source:
                episode['phase'] = self._tensor(source['phase'], 'phase', (steps,), torch.long)
            evidence, schema = self._physical_evidence(source.get('physical_evidence'), steps)
            if schema != self._evidence_schema:
                raise ValueError('physical_evidence does not match recorded keys, shapes and dtypes')
            if evidence is not None:
                episode['physical_evidence'] = evidence
            episodes.append(episode)
        return episodes

    def save(self, path, contract, qualification):
        """Write successful tensors and caller-supplied qualification evidence.

        ``qualified=False`` is valid for diagnostic storage; the imitation
        loader must independently require actual physical/repeat qualification.
        """
        episodes = self._validated_episodes()
        if not isinstance(contract, Mapping) or not isinstance(qualification, Mapping):
            raise ValueError('contract and qualification must be mappings')
        artifact = {
            'version': 'dh116-tabletop-demonstrations-v1',
            'observation_dim': self.observation_dim,
            'action_dim': self.action_dim,
            'contract': dict(contract),
            'qualification': dict(qualification),
            'episodes': episodes,
        }
        torch.save(artifact, Path(path))
