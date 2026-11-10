"""Independent distribution oracles; no inverse of rounded bounded actions."""
import importlib
import importlib.util

import pytest
import torch
from torch.distributions import Normal, TransformedDistribution, TanhTransform, kl_divergence


def gaussian():
    name = 'go2_pvcnn.control.dh116.tabletop_squashed_gaussian'
    assert importlib.util.find_spec(name), 'separate tanh-latent distribution is missing'
    return importlib.import_module(name)


def test_log_probability_matches_independent_transformed_distribution():
    g = gaussian()
    mu = torch.tensor([[.2, -.4], [.6, -.2]], dtype=torch.float64)
    std = torch.tensor([[.3, .7], [.4, .2]], dtype=torch.float64)
    z = torch.tensor([[.1, -.9], [1.2, .5]], dtype=torch.float64)
    oracle = TransformedDistribution(Normal(mu, std), [TanhTransform()])
    torch.testing.assert_close(g.squashed_log_prob(z, mu, std), oracle.log_prob(z.tanh()).sum(-1))
    changed_mu, changed_std = mu + .17, std * 1.3
    ratio = (g.squashed_log_prob(z, changed_mu, changed_std)-g.squashed_log_prob(z, mu, std)).exp()
    expected = (Normal(changed_mu, changed_std).log_prob(z)-Normal(mu, std).log_prob(z)).sum(-1).exp()
    torch.testing.assert_close(ratio, expected)
    torch.testing.assert_close(g.latent_gaussian_kl(mu, std, changed_mu, changed_std),
                               kl_divergence(Normal(mu, std), Normal(changed_mu, changed_std)).sum(-1))
    assert torch.equal(g.latent_gaussian_kl(mu, std, mu, std), torch.zeros(2, dtype=mu.dtype))


def test_bounded_entropy_has_reparameterized_mu_and_log_std_gradients():
    g = gaussian()
    mu = torch.tensor([[.3, -.7]], dtype=torch.float64, requires_grad=True)
    log_std = torch.tensor([[-.4, -.2]], dtype=torch.float64, requires_grad=True)
    eps = torch.tensor([[.4, -1.2]], dtype=torch.float64)
    actual = g.squashed_entropy(mu, log_std.exp(), eps)
    z = mu + log_std.exp()*eps
    oracle = (Normal(mu, log_std.exp()).entropy() + torch.log1p(-z.tanh().square())).sum(-1)
    torch.testing.assert_close(actual, oracle)
    actual_grad = torch.autograd.grad(actual.sum(), (mu, log_std), retain_graph=True)
    oracle_grad = torch.autograd.grad(oracle.sum(), (mu, log_std))
    for a, b in zip(actual_grad, oracle_grad):
        torch.testing.assert_close(a, b)
        assert torch.count_nonzero(a) == a.numel()


def test_extreme_finite_latents_keep_finite_likelihood_and_gradients():
    g = gaussian()
    z = torch.tensor([[-100., -20., -10., 10., 20., 100.]], requires_grad=True)
    mu = torch.zeros_like(z, requires_grad=True)
    std = torch.ones_like(z, requires_grad=True)
    assert (z.tanh().abs() <= 1).all() and (z.tanh().abs() == 1).any()
    logp = g.squashed_log_prob(z, mu, std)
    assert torch.isfinite(logp).all()
    for grad in torch.autograd.grad(logp.sum(), (z, mu, std)):
        assert torch.isfinite(grad).all()


@pytest.mark.parametrize('field,value', [('z', float('nan')), ('mu', float('inf')),
                                         ('std', 0.), ('std', -1.), ('std', float('inf'))])
def test_nonfinite_or_invalid_distribution_rejected(field, value):
    g = gaussian()
    data = dict(z=torch.zeros(2, 3), mu=torch.zeros(2, 3), std=torch.ones(2, 3))
    data[field][0, 0] = value
    with pytest.raises(ValueError, match='finite|positive'):
        g.squashed_log_prob(**data)
