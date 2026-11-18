"""Visual exact-resume state must be loadable, complete and numerically valid."""
from collections import OrderedDict
import copy
import importlib
import importlib.util
import math

import pytest
import torch

from test_vision_action_derivation import assert_same, source_checkpoint


def state_validator():
    name = 'go2_pvcnn.control.dh116.tabletop_vision_checkpoint_state'
    assert importlib.util.find_spec(name), 'shared complete visual checkpoint state validator is missing'
    return importlib.import_module(name).validate_visual_checkpoint_state


@pytest.fixture
def saved(tmp_path):
    return source_checkpoint(tmp_path)[1]


def corrupt_state(saved, case):
    """Independent corruptions of actual saved policy/Adam/RNG state."""
    if case == 'missing_weight': del saved['policy']['critic.4.bias']
    elif case == 'extra_weight': saved['policy']['actor.extra'] = torch.zeros(1)
    elif case == 'weight_shape': saved['policy']['actor.point_net.0.weight'] = torch.zeros(32, 4)
    elif case == 'weight_dtype': saved['policy']['critic.4.bias'] = saved['policy']['critic.4.bias'].double()
    elif case == 'noise_below_min': saved['policy']['log_std'][0] = math.log(.00099)
    elif case == 'noise_above_max': saved['policy']['log_std'][0] = math.log(.30001)
    elif case == 'noise_nonfinite': saved['policy']['log_std'][0] = float('nan')
    elif case == 'noise_underflow': saved['policy']['log_std'][0] = -1000.
    elif case == 'empty_adam': saved['optimizer']['state'].clear()
    elif case == 'missing_adam_parameter': del saved['optimizer']['state'][18]
    elif case == 'empty_adam_parameter': saved['optimizer']['state'][0] = {}
    elif case == 'missing_adam_moment': del saved['optimizer']['state'][0]['exp_avg']
    elif case == 'moment_shape': saved['optimizer']['state'][0]['exp_avg'] = torch.zeros(25)
    elif case == 'moment_dtype': saved['optimizer']['state'][0]['exp_avg'] = torch.zeros(24, dtype=torch.float64)
    elif case == 'negative_second_moment': saved['optimizer']['state'][0]['exp_avg_sq'][0] = -.001
    elif case == 'negative_max_moment':
        saved['optimizer']['state'][0]['max_exp_avg_sq'] = torch.full((24,), -.001)
    elif case == 'amsgrad_missing_max': saved['optimizer']['param_groups'][0]['amsgrad'] = True
    elif case == 'mapping_reordered': saved['optimizer']['param_groups'][0]['params'].reverse()
    elif case == 'mapping_duplicate': saved['optimizer']['param_groups'][0]['params'][1] = 0
    elif case == 'mapping_float': saved['optimizer']['param_groups'][0]['params'][1] = 1.
    elif case == 'mapping_bool': saved['optimizer']['param_groups'][0]['params'][0] = False
    elif case == 'mapping_unknown_state': saved['optimizer']['state'][19] = copy.deepcopy(saved['optimizer']['state'][0])
    elif case == 'mapping_wrong_names':
        saved['optimizer']['param_groups'][0]['param_names'] = list(reversed(saved['policy']))
    elif case == 'step_vector': saved['optimizer']['state'][0]['step'] = torch.tensor([1.])
    elif case == 'step_negative': saved['optimizer']['state'][0]['step'] = torch.tensor(-1.)
    elif case == 'step_nan': saved['optimizer']['state'][0]['step'] = torch.tensor(float('nan'))
    elif case == 'step_inf': saved['optimizer']['state'][0]['step'] = float('inf')
    elif case == 'step_bool': saved['optimizer']['state'][0]['step'] = True
    elif case == 'step_bool_tensor': saved['optimizer']['state'][0]['step'] = torch.tensor(True)
    elif case == 'step_complex': saved['optimizer']['state'][0]['step'] = torch.tensor(1+0j)
    elif case == 'rng_one_byte': saved['torch_rng'] = torch.tensor([0], dtype=torch.uint8)
    elif case == 'rng_wrong_dtype': saved['torch_rng'] = saved['torch_rng'].long()
    elif case == 'rng_matrix': saved['torch_rng'] = saved['torch_rng'].reshape(1, -1)
    elif case == 'iteration_negative': saved['iteration'] = -1
    elif case == 'iteration_bool': saved['iteration'] = True
    else: raise AssertionError(case)


INVALID_CASES = (
    'missing_weight', 'extra_weight', 'weight_shape', 'weight_dtype',
    'noise_below_min', 'noise_above_max', 'noise_nonfinite', 'noise_underflow',
    'empty_adam', 'missing_adam_parameter', 'empty_adam_parameter',
    'missing_adam_moment', 'moment_shape', 'moment_dtype', 'negative_second_moment',
    'negative_max_moment', 'amsgrad_missing_max', 'mapping_reordered',
    'mapping_duplicate', 'mapping_float', 'mapping_bool', 'mapping_unknown_state',
    'mapping_wrong_names', 'step_vector', 'step_negative', 'step_nan', 'step_inf',
    'step_bool', 'step_bool_tensor', 'step_complex', 'rng_one_byte',
    'rng_wrong_dtype', 'rng_matrix', 'iteration_negative', 'iteration_bool',
)


def test_real_complete_state_is_accepted_without_mutating_input_or_global_rng(saved):
    validate = state_validator()
    before = copy.deepcopy(saved)
    rng = torch.get_rng_state().clone()
    validate(saved)
    assert_same(saved, before)
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)


def test_weight_dictionary_order_is_not_policy_semantics(saved):
    validate = state_validator()
    saved['policy'] = OrderedDict(reversed(list(saved['policy'].items())))
    validate(saved)


def test_iteration_zero_allows_pristine_adam_state(saved):
    validate = state_validator()
    saved['iteration'] = 0
    saved['timesteps'] = 0
    saved['optimizer']['state'].clear()
    validate(saved)


def test_fp32_projected_log_noise_endpoints_are_accepted(saved):
    validate = state_validator()
    saved['policy']['log_std'][::2] = math.log(.001)
    saved['policy']['log_std'][1::2] = math.log(.3)
    validate(saved)


def test_positive_real_amsgrad_and_python_scalar_steps_are_accepted(saved):
    validate = state_validator()
    saved['optimizer']['param_groups'][0]['amsgrad'] = True
    for entry in saved['optimizer']['state'].values():
        entry['max_exp_avg_sq'] = entry['exp_avg_sq'].clone()
        entry['step'] = 1.
    validate(saved)


@pytest.mark.parametrize('case', INVALID_CASES)
def test_corrupted_state_is_rejected_without_changing_rng_or_input(saved, case):
    validate = state_validator()
    corrupt_state(saved, case)
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError): validate(saved)
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)


@pytest.mark.parametrize('name', ['lr', 'betas', 'eps', 'weight_decay'])
def test_adam_group_requires_each_saved_hyperparameter(saved, name):
    validate = state_validator()
    del saved['optimizer']['param_groups'][0][name]
    with pytest.raises(ValueError): validate(saved)


@pytest.mark.parametrize('name,value', [
    ('lr', -.001), ('lr', float('nan')), ('lr', float('inf')),
    ('lr', '0.000417'), ('lr', True), ('lr', torch.tensor(.000417)),
    ('eps', -1e-8), ('eps', float('nan')), ('eps', float('inf')),
    ('eps', '1e-8'), ('eps', True),
    ('weight_decay', -.001), ('weight_decay', float('nan')),
    ('weight_decay', float('inf')), ('weight_decay', '0'), ('weight_decay', False),
    ('betas', None), ('betas', (.9,)), ('betas', (.9, .999, .8)),
    ('betas', (0, .999)), ('betas', (True, .999)), ('betas', ('0.9', .999)),
    ('betas', (-.01, .999)), ('betas', (.9, -.01)),
    ('betas', (1., .999)), ('betas', (.9, 1.)),
    ('betas', (float('nan'), .999)), ('betas', (.9, float('inf'))),
])
def test_adam_group_rejects_invalid_types_nonfinite_values_and_ranges(saved, name, value):
    validate = state_validator()
    saved['optimizer']['param_groups'][0][name] = value
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError): validate(saved)
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)


def test_adam_group_learning_rate_must_equal_saved_adaptive_learning_rate(saved):
    validate = state_validator()
    saved['optimizer']['param_groups'][0]['lr'] = saved['learning_rate']*2
    with pytest.raises(ValueError, match='learning.rate|lr'): validate(saved)


@pytest.mark.parametrize('name', ['amsgrad', 'maximize', 'capturable', 'differentiable',
                                'decoupled_weight_decay'])
@pytest.mark.parametrize('value', [None, 0, 'False'])
def test_adam_boolean_flags_reject_nonboolean_values_when_present(saved, name, value):
    validate = state_validator()
    saved['optimizer']['param_groups'][0][name] = value
    with pytest.raises(ValueError): validate(saved)


@pytest.mark.parametrize('name', ['foreach', 'fused'])
@pytest.mark.parametrize('value', [0, 'False'])
def test_adam_optional_boolean_flags_reject_nonboolean_values(saved, name, value):
    validate = state_validator()
    saved['optimizer']['param_groups'][0][name] = value
    with pytest.raises(ValueError): validate(saved)


@pytest.mark.parametrize('betas', [(.0, .999), [.9, .999]])
def test_adam_accepts_legal_scalar_hyperparameters_and_unknown_extensions(saved, betas):
    validate = state_validator()
    saved['learning_rate'] = .00038443359375000017
    group = saved['optimizer']['param_groups'][0]
    group.update(lr=saved['learning_rate'], betas=betas, eps=0., weight_decay=0,
        amsgrad=False, maximize=False, capturable=False, differentiable=False,
        decoupled_weight_decay=False, foreach=None, fused=None,
        vendor_positive_extension=1.25)
    before = copy.deepcopy(saved)
    rng = torch.get_rng_state().clone()
    validate(saved)
    assert_same(saved, before)
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)


def test_adam_optional_flags_may_be_absent_for_torch_load_defaults(saved):
    validate = state_validator()
    group = saved['optimizer']['param_groups'][0]
    for key in ('amsgrad', 'maximize', 'capturable', 'differentiable',
                'decoupled_weight_decay', 'foreach', 'fused'):
        group.pop(key, None)
    validate(saved)


def test_ppo_minimization_rejects_an_adam_ascent_mode(saved):
    # Independent real-optimizer oracle: a positive loss gradient must lower
    # the parameter. maximize=True reverses that direction, despite valid state.
    descent = torch.nn.Parameter(torch.tensor(1.))
    ascent = torch.nn.Parameter(torch.tensor(1.))
    descent.grad = torch.tensor(1.)
    ascent.grad = torch.tensor(1.)
    torch.optim.Adam([descent], lr=.01).step()
    torch.optim.Adam([ascent], lr=.01, maximize=True).step()
    assert descent.item() < 1. < ascent.item()
    saved['optimizer']['param_groups'][0]['maximize'] = True
    with pytest.raises(ValueError, match='maximize|minimiz'):
        state_validator()(saved)


@pytest.mark.parametrize('conflict', [dict(foreach=True), dict(differentiable=True)])
def test_adam_rejects_torch27_incompatible_fused_flag_combinations(saved, conflict):
    validate = state_validator()
    saved['optimizer']['param_groups'][0].update(fused=True, **conflict)
    with pytest.raises(ValueError): validate(saved)
