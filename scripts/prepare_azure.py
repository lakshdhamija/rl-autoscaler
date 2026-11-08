"""Build the processed real-traffic dataset from the Azure Functions 2019 trace.

Download and extract the trace into data/raw/azure2019 first (about 143 MB compressed, 2 GB extracted):
  https://github.com/Azure/AzurePublicDataset/releases/download/dataset-functions-2019/azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz

Usage: python scripts/prepare_azure.py [--raw data/raw/azure2019]

Writes data/azure2019_http_apps.csv.gz (requests per minute for each selected app, 14 days)
and data/azure2019_http_apps.json (source, license, selection rule, train/test split).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MINUTES = [str(i) for i in range(1, 1441)]
DAYS = 14
TEST_APPS = 5


def http_apps(raw):
    """Sum invocations of HTTP-triggered functions per app, for every minute of the 14 days."""
    per_day = []
    for day in range(1, DAYS + 1):
        df = pd.read_csv(raw / f"invocations_per_function_md.anon.d{day:02d}.csv", dtype={m: "int32" for m in MINUTES})
        per_day.append(df[df.Trigger == "http"].groupby("HashApp")[MINUTES].sum())
    apps = sorted(set().union(*(set(d.index) for d in per_day)))
    series = pd.DataFrame(0.0, index=apps, columns=range(DAYS * 1440))
    for day, d in enumerate(per_day):  # an app missing from a day's file had no requests that day
        series.loc[d.index, day * 1440:(day + 1) * 1440 - 1] = d.to_numpy()
    return series


def select(series):
    """Keep busy apps that serve traffic every day and actually vary; drop near-duplicate apps."""
    x = series.to_numpy()
    mean = x.mean(1)
    keep = ((x.reshape(len(x), DAYS, 1440).sum(2) > 0).all(1)  # traffic on all 14 days
            & (mean >= 500)                                   # at least 500 requests per minute on average
            & (x.std(1) / mean >= 0.15)                       # not flat: nothing to autoscale
            & ((x == 0).mean(1) < 0.05))                      # not an on/off batch job
    chosen = series[keep]
    # Some owners run copies of one app; a copy would leak training traffic into the test set.
    corr = np.corrcoef(chosen.to_numpy())
    unique = [i for i in range(len(chosen)) if not any(corr[i, j] > 0.95 for j in range(i))]
    return chosen.iloc[unique]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw/azure2019", type=Path)
    parser.add_argument("--out", default="data/azure2019_http_apps", type=Path)
    args = parser.parse_args()

    apps = select(http_apps(args.raw))
    apps.index = [a[:8] for a in apps.index]  # short ids; the full hashes are anonymous anyway
    assert apps.index.is_unique
    test = sorted(np.random.default_rng(0).choice(apps.index, TEST_APPS, replace=False))
    train = sorted(set(apps.index) - set(test))

    apps.T.rename_axis("minute").to_csv(args.out.with_suffix(".csv.gz"), float_format="%.0f")
    meta = {
        "source": "Azure Functions Trace 2019, https://github.com/Azure/AzurePublicDataset",
        "license": "CC-BY 4.0",
        "citation": "Shahrad et al., Serverless in the Wild: Characterizing and Optimizing the Serverless "
                    "Workload at a Large Cloud Provider, USENIX ATC 2020",
        "content": "Requests per minute per app (sum of its HTTP-triggered functions), 14 days, minute 0 = day 1 00:00",
        "selection": "HTTP traffic on all 14 days, mean >= 500 requests/minute, coefficient of variation >= 0.15, "
                     "< 5% zero minutes, near-duplicate apps (correlation > 0.95) removed",
        "train_apps": train,
        "test_apps": test,
        "train_days": [1, 10],
        "test_days": [11, 14],
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{len(apps)} apps: {len(train)} train, {len(test)} test -> {args.out}.csv.gz")


if __name__ == "__main__":
    main()
