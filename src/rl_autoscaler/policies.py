"""Baseline policies and an episode runner. A policy reads the env state and returns an action."""

import math
from collections import deque

import numpy as np


def _action(env, desired):
    c = env.config
    total = env.running + len(env.booting)
    return int(np.clip(desired - total, -c.max_step, c.max_step)) + c.max_step


class FixedPolicy:
    """Hold a fixed server count."""

    def __init__(self, servers):
        self.servers = servers

    def reset(self):
        pass

    def act(self, env):
        return _action(env, self.servers)


class ThresholdPolicy:
    """The Kubernetes Horizontal Pod Autoscaler rule: desired = ceil(load / (capacity * target)).

    Keeps the two HPA defaults that change behaviour here: a 10% tolerance band, and a
    5-minute scale-down stabilization window (scale down only to the highest count asked
    for in the last 5 minutes). It reacts to the last observed minute of load.
    """

    def __init__(self, target=0.7, tolerance=0.1, down_window=5):
        self.target, self.tolerance, self.down_window = target, tolerance, down_window

    def reset(self):
        self.recent = deque(maxlen=self.down_window)

    def load_estimate(self, env):
        return env.history[-1]

    def act(self, env):
        total = env.running + len(env.booting)
        raw = self.load_estimate(env) / (env.config.capacity_per_server * self.target)
        desired = total if abs(raw / total - 1) <= self.tolerance else math.ceil(raw)
        self.recent.append(desired)
        if desired < total:
            desired = min(total, max(self.recent))
        return _action(env, desired)


class ForecastPolicy(ThresholdPolicy):
    """The same rule, applied to a linear-trend forecast of load at the minute new servers would be ready."""

    def __init__(self, fit_minutes=5, **kwargs):
        super().__init__(**kwargs)
        self.fit_minutes = fit_minutes

    def load_estimate(self, env):
        y = np.array(env.history)[-self.fit_minutes:]
        slope, intercept = np.polyfit(np.arange(len(y)), y, 1)
        ahead = len(y) + env.config.boot_delay  # last observed minute is len(y) - 1
        return max(0.0, intercept + slope * ahead)


class SeasonalPolicy(ThresholdPolicy):
    """The same rule on a "same time yesterday" forecast: the peak of yesterday's load from now
    until new servers would be ready, adjusted for how busy today is compared with yesterday.
    It never plans for less than the last observed minute. Needs yesterday_window > 0."""

    def load_estimate(self, env):
        t, c = env.t, env.config
        if t < 1440:
            raise ValueError("SeasonalPolicy needs yesterday's traffic (yesterday_window > 0)")
        ahead = env.load[t - 1440:t - 1440 + c.boot_delay + 1]
        before = env.load[max(0, t - 1440 - c.window):t - 1440]
        level = np.clip(np.mean(env.history) / max(np.mean(before), 1.0), 0.5, 2.0) if len(before) else 1.0
        return max(env.history[-1], level * ahead.max())


def rollout(env, policy, seed=None):
    """Run one episode and return the info dict of every step, with the reward added."""
    policy.reset()
    env.reset(seed=seed)
    rows, done = [], False
    while not done:
        _, reward, terminated, truncated, info = env.step(policy.act(env))
        rows.append({**info, "reward": reward})
        done = terminated or truncated
    return rows
