import numpy as np
import pytest

from rl_autoscaler.data import episodes
from rl_autoscaler.env import AutoscaleEnv
from rl_autoscaler.policies import SeasonalPolicy, rollout

pytest.importorskip("stable_baselines3")
from rl_autoscaler.train import PPOPolicy, ResidualEnv, mixed_traffic, train  # noqa: E402


def test_mixed_traffic_draws_both_kinds():
    draw = mixed_traffic()
    rng = np.random.default_rng(0)
    traces = [draw(rng) for _ in range(40)]
    assert all(len(t) == 2880 for t in traces)
    real = {id(t) for t in traces}  # real days are reused objects, synthetic days are new arrays
    assert 1 < len(real) < 40


def test_trained_policy_runs_a_day():
    """Tiny training run: checks the pieces fit together, not that the agent learns."""
    model = train(late_penalty=0.01, seed=0, steps=2048, n_envs=1)
    rows = rollout(AutoscaleEnv(np.full(2880, 6000.0)), PPOPolicy(model))
    assert len(rows) == 1440


def test_residual_with_no_adjustment_is_the_rule():
    """Action 5 (adjust by 0) on every minute must reproduce the "same time yesterday" rule exactly."""

    class NoAdjustment:
        def reset(self):
            pass

        def act(self, env):
            return 5

    trace = episodes("test_days")[0][1]
    rule = rollout(AutoscaleEnv(trace), SeasonalPolicy(target=0.7))
    residual = rollout(ResidualEnv(AutoscaleEnv(trace)), NoAdjustment())
    assert rule == residual


def test_trained_residual_policy_runs_a_day():
    model = train(late_penalty=0.01, seed=0, steps=2048, n_envs=1, residual=True)
    assert model.observation_space.shape == (26,)  # the rule's action is the extra number
    rows = rollout(AutoscaleEnv(np.full(2880, 6000.0)), PPOPolicy(model, residual=True))
    assert len(rows) == 1440
