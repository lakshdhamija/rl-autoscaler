"""Train the PPO agents in parallel on Modal CPUs: 2 kinds x 5 late penalties x 3 seeds = 30 runs.

Kinds: "plain" learns to scale from nothing (10M steps); "residual" learns a correction to the
"same time yesterday" rule (3M steps; it starts at the rule, so it needs fewer).
Each run trains on mixed traffic (real training days and fitted synthetic days, 50/50), then is
scored on the unseen days and the held-out apps. Results go to results/ppo/.

Usage: modal run --detach scripts/modal_train.py [--smoke]  (--smoke: two short runs to test the setup)
Runs save to the Modal volume rl-autoscaler-results, so they finish even if this laptop disconnects.
Fetch them with: modal volume get rl-autoscaler-results ppo results/ --force
"""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
# Cost per late request; one server-minute costs 1. Below about 1/600 (one server's capacity per minute)
# a late request costs less than a server, so the best policy runs ~1 server (a first batch at 0.001 and 0.003 did).
PENALTIES = (0.01, 0.03, 0.1, 0.3, 1.0)
SEEDS = (0, 1, 2)
STEPS = {"plain": 10_000_000, "residual": 3_000_000}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.15.0", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("gymnasium==1.4.0", "numpy==2.4.6", "stable-baselines3==2.9.2")
    .env({"PYTHONPATH": "/repo/src"})
    .add_local_dir(ROOT / "src", "/repo/src", ignore=["**/__pycache__"])
    .add_local_dir(ROOT / "data", "/repo/data", ignore=["raw"])
)
app = modal.App("rl-autoscaler-ppo", image=image)
volume = modal.Volume.from_name("rl-autoscaler-results", create_if_missing=True)


@app.function(cpu=2, timeout=4 * 3600, volumes={"/results": volume})
def train_one(kind: str, penalty: float, seed: int, steps: int):
    import time

    from rl_autoscaler.train import PPOPolicy, score, train

    residual = kind == "residual"
    start = time.time()
    model = train(penalty, seed, steps, residual=residual)
    minutes = (time.time() - start) / 60
    result = {"kind": kind, "penalty": penalty, "seed": seed, "steps": steps, "train_minutes": round(minutes, 1),
              **{split: score(lambda: PPOPolicy(model, residual), split) for split in ("test_days", "test_apps")}}
    out = Path("/results/ppo" if steps >= 1_000_000 else "/results/smoke")  # smoke runs stay out of results/ppo
    out.mkdir(parents=True, exist_ok=True)
    name = f"{kind}-{steps // 1_000_000}M_penalty-{penalty}_seed-{seed}"
    model.save(out / f"{name}.zip")
    (out / f"{name}.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.local_entrypoint()
def main(smoke: bool = False):
    jobs = ([(kind, 0.1, 0, 20_000) for kind in STEPS] if smoke
            else [(kind, p, s, steps) for kind, steps in STEPS.items() for p in PENALTIES for s in SEEDS])
    # return_exceptions: one failed run is reported instead of cancelling the other runs
    for result in train_one.starmap(jobs, return_exceptions=True):
        if isinstance(result, Exception):
            print(f"FAILED: {result!r}")
            continue
        print(f"{result['kind']} penalty-{result['penalty']}_seed-{result['seed']}: {result['train_minutes']} min, held-out apps "
              f"{result['test_apps']['cost_per_day']:,.0f} server-min/day, {result['test_apps']['late_pct']:.2f}% late")
