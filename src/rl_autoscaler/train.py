"""Train a PPO autoscaler (Stable-Baselines3) and run it like any other policy."""

from dataclasses import replace

import gymnasium as gym
import numpy as np

from rl_autoscaler.data import episodes, train_series
from rl_autoscaler.env import AutoscaleEnv, ClusterConfig
from rl_autoscaler.policies import SeasonalPolicy
from rl_autoscaler.traffic import FittedTraffic


def mixed_traffic():
    """Each episode is a real training day or a fitted synthetic day, 50/50.

    Real days keep the agent close to real traffic; synthetic days stop it memorizing
    the 126 real training episodes. Unseen days and held-out apps are never used here.
    """
    real = [trace for _, trace in episodes("train")]
    fitted = FittedTraffic(train_series())

    def draw(rng):
        return real[rng.integers(len(real))] if rng.random() < 0.5 else fitted(rng)

    return draw


class Residual:
    """The "same time yesterday" rule at a 70% target, plus the agent's adjustment of -5..+5 servers to its action.

    Residual policy learning (Silver et al. 2018, Johannink et al. 2019): the agent starts from a
    good rule and learns only the correction, instead of learning to scale from nothing.
    """

    def __init__(self):
        self.rule = SeasonalPolicy(target=0.7)

    def reset(self):
        self.rule.reset()

    def observe(self, env, obs):
        """Ask the rule for its action and add it to the observation, scaled to -1..1."""
        m = env.config.max_step
        self.base = self.rule.act(env)
        return np.append(obs, (self.base - m) / m).astype(np.float32)

    def action(self, env, adjustment):
        m = env.config.max_step
        return int(np.clip(self.base + int(adjustment) - m, 0, 2 * m))


class ResidualEnv(gym.Wrapper):
    """AutoscaleEnv where the agent's action adjusts the rule's action (see Residual)."""

    def __init__(self, env):
        super().__init__(env)
        self.residual = Residual()
        space = env.observation_space
        self.observation_space = gym.spaces.Box(np.append(space.low, -1).astype(np.float32), np.append(space.high, 1).astype(np.float32))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.residual.reset()
        return self.residual.observe(self.env.unwrapped, obs), info

    def step(self, action):
        env = self.env.unwrapped
        obs, reward, terminated, truncated, info = self.env.step(self.residual.action(env, action))
        return self.residual.observe(env, obs), reward, terminated, truncated, info


def train(late_penalty, seed, steps, n_envs=8, residual=False):
    """Train one PPO agent. late_penalty sets the cost against late-requests trade-off.
    residual=True trains a correction to the "same time yesterday" rule instead of a policy from scratch."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize

    torch.set_num_threads(1)  # the network is small; more threads only add overhead
    config = replace(ClusterConfig(), late_penalty=late_penalty)
    traffic = mixed_traffic()
    make = (lambda: ResidualEnv(AutoscaleEnv(traffic, config))) if residual else (lambda: AutoscaleEnv(traffic, config))
    venv = make_vec_env(make, n_envs=n_envs, seed=seed)
    # Observations are already scaled by cluster capacity; only the reward needs scaling for PPO.
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=0.99)
    model = PPO("MlpPolicy", venv, gamma=0.99, seed=seed, verbose=0, device="cpu")
    if residual:  # start with "no adjustment" half the time, so the agent starts close to the rule
        model.policy.action_net.bias.data[config.max_step] = np.log(venv.action_space.n - 1)
    model.learn(total_timesteps=steps)
    return model


def score(make_policy, split):
    """Run a policy on every episode of a test split. Returns server-minutes per day and late/dropped %."""
    from rl_autoscaler.policies import rollout

    traces = episodes(split)
    rows = [r for _, trace in traces for r in rollout(AutoscaleEnv(trace), make_policy())]
    load = sum(r["load"] for r in rows)
    return {"cost_per_day": sum(r["cost"] for r in rows) / len(traces),
            "late_pct": 100 * sum(r["late"] for r in rows) / load,
            "dropped_pct": 100 * sum(r["dropped"] for r in rows) / load}


class PPOPolicy:
    """A trained PPO model with the same reset/act interface as the baselines in policies.py.
    residual=True for a model trained with residual=True."""

    def __init__(self, model, residual=False):
        self.model = model
        self.residual = Residual() if residual else None

    def reset(self):
        if self.residual:
            self.residual.reset()

    def act(self, env):
        obs = env._obs()
        if self.residual is None:
            return int(self.model.predict(obs, deterministic=True)[0])
        adjustment, _ = self.model.predict(self.residual.observe(env, obs), deterministic=True)
        return self.residual.action(env, adjustment)

