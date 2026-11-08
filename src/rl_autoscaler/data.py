"""Real traffic: busy HTTP apps from the Azure Functions 2019 trace (built by scripts/prepare_azure.py)."""

import gzip
import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data" / "azure2019_http_apps"
P99_TARGET = 14_700.0  # busiest 1% of minutes need ~35 of 50 servers at 70% utilization; bursts go higher
TRAIN_MINUTES = 10 * 1440  # days 1-10


def load():
    """Return ({app id: requests per minute over 14 days}, metadata with source, license and split)."""
    with gzip.open(DATA.with_suffix(".csv.gz"), "rt") as f:
        names = f.readline().strip().split(",")[1:]
        values = np.loadtxt(f, delimiter=",", ndmin=2)[:, 1:]
    return dict(zip(names, values.T)), json.loads(DATA.with_suffix(".json").read_text())


def scale(series):
    """Scale an app so its busiest 1% of training minutes sits at P99_TARGET. Uses days 1-10 only."""
    return series * P99_TARGET / np.percentile(series[:TRAIN_MINUTES], 99)


def episodes(split):
    """Two-day traces (yesterday, then the day the agent acts on), scaled to the cluster.

    train:     training apps, days 2-10
    test_days: training apps, days 11-14 (unseen days)
    test_apps: held-out apps, days 2-14 (unseen apps)
    """
    apps, meta = load()
    names, days = {"train": (meta["train_apps"], range(2, 11)),
                   "test_days": (meta["train_apps"], range(11, 15)),
                   "test_apps": (meta["test_apps"], range(2, 15))}[split]
    return [(f"{name} day {d}", scale(apps[name])[(d - 2) * 1440:d * 1440]) for name in names for d in days]


def train_series():
    """Unscaled days 1-10 of every training app, for fitting the synthetic generator."""
    apps, meta = load()
    return [apps[name][:TRAIN_MINUTES] for name in meta["train_apps"]]
