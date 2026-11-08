import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rl_autoscaler.data import P99_TARGET, episodes, load, train_series
from rl_autoscaler.env import AutoscaleEnv, ClusterConfig
from rl_autoscaler.policies import FixedPolicy, ForecastPolicy, SeasonalPolicy, ThresholdPolicy, rollout
from rl_autoscaler.traffic import FittedTraffic, find_bursts

HOLD = ClusterConfig().max_step  # action index for "no change"


@pytest.fixture(scope="module")
def fitted():
    return FittedTraffic(train_series())


def cfg(**kwargs):
    """Config for short hand-made traces: no yesterday observation."""
    return ClusterConfig(yesterday_window=0, **kwargs)


def totals(rows):
    return [r["running"] + r["booting"] for r in rows]


def test_passes_gymnasium_env_checker(fitted):
    check_env(AutoscaleEnv(fitted), skip_render_check=True)


def test_new_server_serves_only_after_boot_delay_but_is_billed_at_once():
    env = AutoscaleEnv(np.full(5, 100.0), cfg(capacity_per_server=100, boot_delay=3))
    env.reset(seed=0)  # starts with 1 server
    rows = [env.step(HOLD + 1)[4]] + [env.step(HOLD)[4] for _ in range(4)]
    assert [r["running"] for r in rows] == [1, 1, 1, 2, 2]
    assert rows[0]["cost"] == 2.0


def test_scale_down_cancels_booting_servers_first():
    env = AutoscaleEnv(np.full(5, 100.0), cfg(capacity_per_server=100))
    env.reset(seed=0)
    env.step(HOLD + 2)
    info = env.step(HOLD - 1)[4]
    assert (info["running"], info["booting"]) == (1, 1)


def test_overload_queues_requests_and_counts_each_late_request_once():
    env = AutoscaleEnv(np.array([100.0, 150, 150, 0, 0]), cfg(capacity_per_server=100, max_servers=1, timeout_minutes=100))
    rows = rollout(env, FixedPolicy(1))
    assert [r["late"] for r in rows] == [0, 50, 100, 0, 0]
    assert [r["backlog"] for r in rows] == [0, 50, 100, 0, 0]


def test_requests_that_would_wait_past_the_timeout_are_dropped():
    env = AutoscaleEnv(np.array([100.0, 400]), cfg(capacity_per_server=100, max_servers=1, timeout_minutes=1))
    rows = rollout(env, FixedPolicy(1))
    # minute 1: 100 served, 300 late; the queue holds 1 minute of capacity (100), so 200 are dropped
    assert (rows[1]["late"], rows[1]["backlog"], rows[1]["dropped"]) == (300, 100, 200)


def test_every_request_is_served_queued_or_dropped(fitted):
    rows = rollout(AutoscaleEnv(fitted), ThresholdPolicy(), seed=1)
    arrived = sum(r["load"] for r in rows)
    accounted = sum(r["served"] + r["dropped"] for r in rows) + rows[-1]["backlog"]
    assert arrived == pytest.approx(accounted)


def test_yesterday_is_history_and_shows_in_the_observation():
    trace = np.arange(2880, dtype=float)
    env = AutoscaleEnv(trace)
    obs, _ = env.reset(seed=0)
    full = env.config.max_servers * env.config.capacity_per_server
    assert env.t == 1440 and list(env.history) == list(trace[1430:1440])
    assert np.allclose(obs[10:20] * full, trace[0:10])  # yesterday from this time on
    rows = rollout(env, ThresholdPolicy())
    assert len(rows) == 1440  # the agent acts on day 2 only


def test_threshold_scales_to_hpa_target_and_waits_five_minutes_to_scale_down():
    load = np.array([4200.0] * 10 + [420.0] * 10)  # 4200 / (600 * 0.7) = 10 servers, then 1
    n = totals(rollout(AutoscaleEnv(load, cfg()), ThresholdPolicy()))
    assert n[0] == 10
    assert n[11:15] == [10, 10, 10, 10]  # the drop is first seen at minute 11
    assert n[15] == 5  # then it scales down, at most max_step per minute


def test_forecast_scales_ahead_of_threshold_on_a_ramp():
    load = np.linspace(600.0, 12000.0, 40)
    late = {name: sum(r["late"] for r in rollout(AutoscaleEnv(load, cfg()), p))
            for name, p in [("threshold", ThresholdPolicy()), ("forecast", ForecastPolicy())]}
    assert late["forecast"] < late["threshold"]


def test_seasonal_adds_servers_before_a_burst_that_happened_yesterday():
    day = np.full(1440, 4200.0)
    day[600:610] = 12600.0  # the same 10-minute burst at 10:00 on both days
    trace = np.concatenate([day, day])
    rows = {name: rollout(AutoscaleEnv(trace), p) for name, p in [("threshold", ThresholdPolicy()), ("seasonal", SeasonalPolicy())]}
    assert rows["seasonal"][600]["running"] > rows["threshold"][600]["running"]
    assert sum(r["late"] for r in rows["seasonal"]) < sum(r["late"] for r in rows["threshold"])


def test_find_bursts_measures_height_length_and_ramp():
    x = np.full(100, 1000.0)
    x[50:53] = [2000.0, 3000.0, 3000.0]
    (start, height, length, ramp), = find_bursts(x)[0]
    assert (start, height, length, ramp) == (50, 3.0, 3, 1)


def test_fitted_traffic_is_seeded_two_days_and_scaled(fitted):
    a, b = fitted(np.random.default_rng(7)), fitted(np.random.default_rng(7))
    assert a.shape == (2880,) and (a >= 0).all() and np.array_equal(a, b)
    assert np.percentile(a, 99) == pytest.approx(P99_TARGET)


def test_real_data_split_keeps_test_apps_and_days_apart():
    apps, meta = load()
    assert not set(meta["train_apps"]) & set(meta["test_apps"])
    assert set(apps) == set(meta["train_apps"]) | set(meta["test_apps"])
    assert all(len(x) == 14 * 1440 for x in apps.values())
    train = {name.split()[0] for name, _ in episodes("train")}
    assert train == set(meta["train_apps"])
    assert all(len(trace) == 2880 for _, trace in episodes("test_apps"))
