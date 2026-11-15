"""Cost against late requests: PPO agents (results/ppo/*.json) against the baselines at several targets.

A policy is better if it has fewer late requests at the same cost, so each method is a curve:
the baselines sweep their target utilization, each PPO kind sweeps its late penalty (mean of 3 seeds, bars = min/max).

Usage: python scripts/frontier.py [--out results/frontier.png]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rl_autoscaler.policies import SeasonalPolicy, ThresholdPolicy
from rl_autoscaler.train import score

SPLITS = {"test_days": "Unseen days (days 11-14 of training apps)", "test_apps": "Unseen apps (5 held-out apps)"}
TARGETS = (0.4, 0.5, 0.6, 0.7, 0.8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/frontier.png")
    args = parser.parse_args()

    baselines = {name: {split: [score(lambda: policy(target=t), split) for t in TARGETS] for split in SPLITS}
                 for name, policy in (("Kubernetes rule (HPA)", ThresholdPolicy), ("same time yesterday", SeasonalPolicy))}
    ppo = defaultdict(lambda: defaultdict(list))  # kind -> penalty -> runs
    for path in sorted(Path("results/ppo").glob("*.json")):
        r = json.loads(path.read_text())
        ppo[r["kind"]][r["penalty"]].append(r)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (split, title) in zip(axes, SPLITS.items()):
        print(f"\n{title}")
        for name, by_split in baselines.items():
            points = by_split[split]
            ax.plot([p["cost_per_day"] for p in points], [p["late_pct"] for p in points], "o-",
                    label=f"{name}, targets {TARGETS[0]:.0%}-{TARGETS[-1]:.0%}")
            for t, p in zip(TARGETS, points):
                print(f"  {name:22} target {t:.0%}   {p['cost_per_day']:8,.0f} server-min/day  {p['late_pct']:5.2f}% late")
        for kind, by_penalty in sorted(ppo.items()):
            label = f"PPO {kind}, {next(iter(by_penalty.values()))[0]['steps'] / 1e6:g}M steps"
            rows = []
            for penalty in sorted(by_penalty):
                cost = np.array([r[split]["cost_per_day"] for r in by_penalty[penalty]])
                late = np.array([r[split]["late_pct"] for r in by_penalty[penalty]])
                rows.append((cost.mean(), late.mean(), late.min(), late.max()))
                print(f"  {label:22} penalty {penalty:<5} {cost.mean():8,.0f} server-min/day  {late.mean():5.2f}% late"
                      f"  (seeds {late.min():.2f}-{late.max():.2f}, n={len(late)})")
            c, m, lo, hi = map(np.array, zip(*rows))
            ax.errorbar(c, m, yerr=[m - lo, hi - m], fmt="s-", capsize=3, label=f"{label}, late penalties (3 seeds)")
        ax.set_title(title)
        ax.set_xlabel("server-minutes per day (cost)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("late requests (%)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"\nplot saved to {args.out}")


if __name__ == "__main__":
    main()
