"""Autoscaling environment. One step is one simulated minute."""

import math
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np


@dataclass
class ClusterConfig:
    """Every number that defines the simulated cluster. Change these to run a different scenario."""

    capacity_per_server: float = 600.0  # requests per minute one server can serve (no CPU or memory model)
    boot_delay: int = 3                 # minutes before a new server serves traffic
    min_servers: int = 1
    max_servers: int = 50
    max_step: int = 5                   # largest change in server count per minute
    timeout_minutes: float = 2.0        # requests that would wait longer than this are dropped
    server_cost: float = 1.0            # cost per server-minute; booting servers are billed too
    late_penalty: float = 0.01          # cost per request not served in the minute it arrived
    window: int = 10                    # minutes of past load in the observation
    yesterday_window: int = 10          # minutes of yesterday's load from this time on (0 = off); needs a 2-day trace


class AutoscaleEnv(gym.Env):
    """A cluster of identical servers in front of a request queue.

    Each minute the agent adds or removes servers, then that minute's requests arrive.
    Capacity serves the oldest queued requests first (FIFO); requests not served in the
    minute they arrive count as late and wait in the queue.

    traffic: a 1-D array of requests per minute, or a function rng -> array that builds
    a new trace on every reset. Minute 0 of a trace is taken as midnight. When
    yesterday_window > 0 the first day of the trace is history only: the agent acts on the rest.
    """

    metadata = {"render_modes": []}

    def __init__(self, traffic, config=None):
        self.traffic = traffic
        self.config = config or ClusterConfig()
        c = self.config
        # The possible actions: integers 0..10. Action minus 5 is the change in servers,
        # so 0 = remove 5, 5 = no change, 10 = add 5.
        self.action_space = gym.spaces.Discrete(2 * c.max_step + 1)
        # What the agent sees: 25 numbers by default, built in _obs() below.
        self.observation_space = gym.spaces.Box(-1.0, np.inf, shape=(c.window + c.yesterday_window + 5,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        """Start a new episode (one traffic trace, usually one day). Returns the first observation."""
        super().reset(seed=seed)  # seeds self.np_random, so the same seed gives the same day
        c = self.config
        trace = self.traffic(self.np_random) if callable(self.traffic) else self.traffic
        self.load = np.asarray(trace, dtype=np.float64)  # requests arriving in each minute
        self.t = 1440 if c.yesterday_window else 0  # current minute; day 1 is history when yesterday is observed
        if len(self.load) <= self.t:
            raise ValueError("with yesterday_window > 0 the trace needs yesterday plus at least one minute")
        # Start with just enough servers for the last load seen.
        last = self.load[max(self.t - 1, 0)]
        self.running = int(np.clip(math.ceil(last / c.capacity_per_server), c.min_servers, c.max_servers))
        self.booting = []  # minutes left before each booting server is ready
        self.backlog = 0.0  # requests waiting in the queue
        # Load of the last `window` minutes. Without history, assume the first minute's level (warm start).
        past = self.load[max(self.t - c.window, 0):self.t]
        self.history = deque(past if len(past) == c.window else [self.load[0]] * c.window, maxlen=c.window)
        return self._obs(), self._info(0.0, 0.0, 0.0, 0.0, 0.0)

    def step(self, action):
        """Simulate one minute. Returns (observation, reward, terminated, truncated, info)."""
        c = self.config
        # 1. Apply the agent's action: start booting new servers, or remove servers.
        self._scale(int(action) - c.max_step)
        # 2. Servers that finished booting start serving.
        self.running += sum(1 for m in self.booting if m <= 0)
        self.booting = [m for m in self.booting if m > 0]

        # 3. This minute's requests arrive. Serve the queue first, then the new requests.
        arrivals = self.load[self.t]
        capacity = self.running * c.capacity_per_server
        served_old = min(self.backlog, capacity)
        served_new = min(arrivals, capacity - served_old)
        # 4. New requests that were not served are late; they join the queue.
        late = arrivals - served_new
        self.backlog += late - served_old
        # 5. Drop what the queue cannot clear within the timeout.
        dropped = max(0.0, self.backlog - capacity * c.timeout_minutes)
        self.backlog -= dropped

        # 6. Reward: minus the cost of every billed server, minus the penalty for late requests.
        cost = c.server_cost * (self.running + len(self.booting))
        reward = -(cost + c.late_penalty * late)

        # 7. Move the clock one minute.
        self.booting = [m - 1 for m in self.booting]
        self.history.append(arrivals)
        self.t += 1
        truncated = self.t >= len(self.load)  # the trace has ended; the episode stops
        info = self._info(arrivals, served_old + served_new, late, dropped, cost)
        return self._obs(), reward, False, truncated, info

    def _scale(self, delta):
        """Change the total server count by delta, kept between min_servers and max_servers."""
        c = self.config
        total = self.running + len(self.booting)
        target = int(np.clip(total + delta, c.min_servers, c.max_servers))
        self.booting += [c.boot_delay] * max(0, target - total)
        remove = max(0, total - target)
        while remove and self.booting:  # cancel booting servers first: they serve nothing yet
            self.booting.pop()
            remove -= 1
        self.running -= remove

    def _obs(self):
        """The numbers the agent sees, scaled to roughly 0..1 so learning is stable."""
        c = self.config
        full = c.max_servers * c.capacity_per_server  # load that would need every server
        angle = 2 * math.pi * (self.t % 1440) / 1440
        y = self.t - 1440
        yesterday = self.load[y:y + c.yesterday_window] if c.yesterday_window else []
        return np.array(
            [*(np.array(self.history) / full),     # 10 numbers: load in each of the last 10 minutes
             *(np.asarray(yesterday) / full),      # 10 numbers: load yesterday from this time on (repeating bursts)
             self.running / c.max_servers,         # servers serving now
             len(self.booting) / c.max_servers,    # servers still booting
             self.backlog / full,                  # requests waiting in the queue
             math.sin(angle),                      # time of day as a point on a circle,
             math.cos(angle)],                     # so 23:59 and 00:00 are close together
            dtype=np.float32,
        )

    def _info(self, load, served, late, dropped, cost):
        """Raw numbers for this minute, for plots and metrics. The agent does not see these."""
        return {"load": load, "served": served, "late": late, "dropped": dropped, "cost": cost,
                "running": self.running, "booting": len(self.booting), "backlog": self.backlog}
