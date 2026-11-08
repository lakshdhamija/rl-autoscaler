"""Compare the baselines on fitted synthetic traffic and on unseen real traffic, and plot one real day.

Usage: python scripts/demo.py [--synthetic 40] [--out results/demo.png]
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rl_autoscaler.data import episodes, train_series
from rl_autoscaler.env import AutoscaleEnv
from rl_autoscaler.policies import FixedPolicy, ForecastPolicy, SeasonalPolicy, ThresholdPolicy, rollout
from rl_autoscaler.traffic import FittedTraffic, find_bursts, measure

POLICIES = {
    "threshold (Kubernetes HPA)": ThresholdPolicy,
    "linear forecast": ForecastPolicy,
    "same time yesterday": SeasonalPolicy,
}


def evaluate(traces, capacity):
    """Run every policy on every two-day trace. Returns {policy: [rows of each episode]}."""
    runs = {"fixed (sized for the day's peak)": []}
    for _, trace in traces:
        peak = math.ceil(trace[1440:].max() / capacity)  # a reference that knows the future
        runs["fixed (sized for the day's peak)"].append(rollout(AutoscaleEnv(trace), FixedPolicy(peak)))
    for name, policy in POLICIES.items():
        runs[name] = [rollout(AutoscaleEnv(trace), policy()) for _, trace in traces]
    return runs


def summary(runs, days):
    for name, eps in runs.items():
        rows = [r for e in eps for r in e]
        load = sum(r["load"] for r in rows)
        cost = sum(r["cost"] for r in rows) / days
        late = 100 * sum(r["late"] for r in rows) / load
        dropped = 100 * sum(r["dropped"] for r in rows) / load
        print(f"  {name:34} {cost:12.0f} {late:8.2f} {dropped:10.2f}")


def realism(name, traces):
    events = np.concatenate([find_bursts(x)[0] for x in traces])
    noise = np.median([measure(x)["noise"] for x in traces])
    days = sum(len(x) for x in traces) / 1440
    print(f"  {name:22} {noise:6.2f} {len(events) / days:11.1f} {np.median(events[:, 1]):11.2f} "
          f"{np.percentile(events[:, 1], 90):11.2f} {100 * (events[:, 3] == 0).mean():16.0f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", type=int, default=40, help="number of synthetic two-day episodes")
    parser.add_argument("--out", default="results/demo.png")
    args = parser.parse_args()

    real_train = train_series()
    fitted = FittedTraffic(real_train)
    datasets = {
        "fitted synthetic": [(f"seed {s}", fitted(np.random.default_rng(s))) for s in range(args.synthetic)],
        "real, unseen days (days 11-14 of training apps)": episodes("test_days"),
        "real, unseen apps (5 held-out apps)": episodes("test_apps"),
    }

    print("Does the synthetic traffic look like the real traffic?")
    print(f"  {'':22} {'noise':>6} {'bursts/day':>11} {'height p50':>11} {'height p90':>11} {'peak in 1st min':>17}")
    realism("real (training days)", real_train)
    realism("fitted synthetic", [t for _, t in datasets["fitted synthetic"]])
    realism("real (held-out apps)", [t for _, t in datasets["real, unseen apps (5 held-out apps)"]])

    capacity = AutoscaleEnv(np.zeros(2880)).config.capacity_per_server
    all_runs = {}
    for name, traces in datasets.items():
        all_runs[name] = runs = evaluate(traces, capacity)
        print(f"\n{name}: {len(traces)} days, boot delay 3 min")
        print(f"  {'policy':34} {'server-min/day':>12} {'late %':>8} {'dropped %':>10}")
        summary(runs, len(traces))

    # A policy that keeps more spare servers is late less often, so compare at several utilization targets:
    # a better policy has fewer late requests at the same cost.
    for name in ["real, unseen days (days 11-14 of training apps)", "real, unseen apps (5 held-out apps)"]:
        print(f"\ncost against late requests, {name}")
        for label, policy in [("threshold", ThresholdPolicy), ("same time yesterday", SeasonalPolicy)]:
            for target in (0.5, 0.6, 0.7):
                rows = [r for _, trace in datasets[name] for r in rollout(AutoscaleEnv(trace), policy(target=target))]
                load = sum(r["load"] for r in rows)
                print(f"  {label:20} target {target:.0%}  {sum(r['cost'] for r in rows) / len(datasets[name]):8.0f} server-min/day"
                      f"  {100 * sum(r['late'] for r in rows) / load:6.2f}% late")

    # Plot the held-out day where "same time yesterday" cut late requests the most versus the threshold rule.
    name = "real, unseen apps (5 held-out apps)"
    traces, runs = datasets[name], all_runs[name]
    gain = [sum(r["late"] for r in a) - sum(r["late"] for r in b)
            for a, b in zip(runs["threshold (Kubernetes HPA)"], runs["same time yesterday"])]
    i = int(np.argmax(gain))
    label, trace = traces[i]
    today, yesterday = trace[1440:], trace[:1440]
    mid = int(np.argmax(today))
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    for ax, part, title in [(axes[0], slice(None), f"Held-out app {label}"),
                            (axes[1], slice(max(0, mid - 30), mid + 30), "Around the largest burst")]:
        minutes = np.arange(1440)[part]
        ax.plot(minutes, yesterday[part], color="0.7", lw=1, label="requests per minute, yesterday")
        ax.plot(minutes, today[part], color="black", lw=1.2, label="requests per minute, today")
        for policy in ["threshold (Kubernetes HPA)", "same time yesterday"]:
            capacity_line = np.array([r["running"] * capacity for r in runs[policy][i]])
            ax.step(minutes, capacity_line[part], where="post", lw=1.2, label=f"serving capacity, {policy}")
        ax.set_title(title)
        ax.set_xlabel("minute of day")
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"\nplot ({label}, the held-out day where 'same time yesterday' helped most) saved to {args.out}")


if __name__ == "__main__":
    main()
