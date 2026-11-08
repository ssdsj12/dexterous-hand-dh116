"""Explicit simulation-only SDK adapter and coordination experiment hooks."""
import json
from pathlib import Path

import numpy as np
import torch


def install_coordination(env, world, mode, output):
    from .tabletop_bimanual_coordination import BimanualCoordinator, coordination_state_from_native, CONTRACT
    coordinator = BimanualCoordinator(world.n, mode=mode.upper())
    if getattr(env, 'bimanual_action_adapter', None) is not None:
        raise ValueError('only one experimental action adapter may be installed')

    def project(action):
        state = coordination_state_from_native(world, env.task, env)
        physical, diagnostics = coordinator.project(action, state, dt_s=world.dt)
        cancel = diagnostics['cancel_upward_target'] & diagnostics['hold_offset_valid']
        hold_z = diagnostics['hold_offset_m'][:, :, 2]
        world.palms[:, :, 2] = torch.where(cancel,
            torch.minimum(world.palms[:, :, 2], hold_z), world.palms[:, :, 2])
        # The original free-wrist control and collision projection run next.
        env.stage5_executed_action = physical.detach().clone()
        env.stage5_coordination_diagnostics = diagnostics
        return physical

    env.bimanual_action_adapter = project
    with (Path(output)/'coordination_contract.json').open('x') as stream:
        json.dump(dict(CONTRACT, mode=mode.upper(), physics_hz=1/world.dt,
                       student_action_source='M0 actor', execution_source='simulation_coordinator'), stream)


def install_explicit_sdk_adapter(env, world, output):
    from .tabletop_bimanual_physical_action_adapter import HandPhysicalKinematics, BimanualPhysicalActionAdapter
    if getattr(env, 'bimanual_action_adapter', None) is not None:
        raise ValueError('explicit SDK adapter requires an independent action path')
    asset_root = Path(__file__).resolve().parents[4]/'assets/fixed_dual_panda_dh116_v1/dependencies'
    models = {side: HandPhysicalKinematics.from_usd(next(asset_root.glob(f'*_DH116_{side}.usd')), side)
              for side in ('left', 'right')}
    adapter = BimanualPhysicalActionAdapter(models)
    # Explicit experiment derivation, independent of an original checkpoint.
    contract = dict(physical_action_adapter=adapter.contract(),
                    source='explicit_Teacher_physical_action_regression;no_model_overwrite')
    adapter.require_checkpoint_contract(contract)
    max_difference = [0.]
    def project(action):
        poses = [hand.data.body_link_pose_w.detach().cpu().numpy() for hand in world.hands]
        base_ids = [hand.body_names.index('base_link') for hand in world.hands]
        palm_ids = [hand.body_names.index('palm_link') if 'palm_link' in hand.body_names else base_ids[h]
                    for h,hand in enumerate(world.hands)]
        raw = action.detach().cpu().numpy()
        converted = []
        for i in range(world.n):
            base = {side: poses[h][i,base_ids[h]] for h,side in enumerate(('left','right'))}
            palm = {side: poses[h][i,palm_ids[h]] for h,side in enumerate(('left','right'))}
            decoded = adapter.decode(raw[i], base, palm,
                world.wrist_angles[i].detach().cpu().numpy(),
                wrist_seed_poses_w={side:world.root_seed[i,h].detach().cpu().numpy()
                                    for h,side in enumerate(('left','right'))})
            converted.append(adapter.to_sdk(decoded))
        physical = torch.as_tensor(np.stack(converted), dtype=action.dtype, device=action.device)
        max_difference[0] = max(max_difference[0], float((physical-action).abs().max()))
        if max_difference[0] > 1e-7:
            raise RuntimeError('physical SDK semantic roundtrip changed Teacher actions')
        return physical
    env.bimanual_action_adapter = project
    with (Path(output)/'physical_action_adapter_contract.json').open('x') as stream:
        json.dump(contract, stream, indent=2)
    return max_difference
