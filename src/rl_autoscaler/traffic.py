"""Synthetic traffic whose every number is measured from real apps."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from rl_autoscaler.data import P99_TARGET

BURST = 1.5  # a minute is a burst if its load is above 1.5 x the median of the previous 30 minutes
REPEAT_WINDOW = 5  # a burst "repeats" if another day has a burst within 5 minutes of the same time


def trailing_median(x, k=30):
    """Median of the k minutes before each minute (what a live system could know)."""
    padded = np.pad(x, (k, 0), mode="edge")
    return np.median(sliding_window_view(padded, k)[:-1], axis=1)


def find_bursts(x):
    """Return (start minute, height, length, minutes to peak) for each burst in x."""
    base = np.maximum(trailing_median(x), 1.0)
    hot = x > BURST * base
    events, t = [], 0
    while t < len(x):
        if not hot[t]:
            t += 1
            continue
        end = t
        while end + 1 < len(x) and (hot[end + 1] or (end + 2 < len(x) and hot[end + 2])):  # allow one calm minute inside
            end += 1
        seg = x[t:end + 1]
        events.append((t, seg.max() / base[t], end - t + 1, int(np.argmax(seg))))
        t = end + 1
    return np.array(events).reshape(-1, 4), base, hot


def repeat_share(starts, days, rng):
    """Share of bursts that repeat on another day, above what random timing would give (0 to 1)."""
    minute, day = starts % 1440, starts // 1440

    def share(m):
        return np.mean([np.any((day != d) & (np.abs(m - mi) <= REPEAT_WINDOW)) for mi, d in zip(m, day)])

    if len(starts) < 2:
        return 0.0
    chance = np.mean([share(rng.integers(0, 1440, len(starts))) for _ in range(3)])
    return float(np.clip((share(minute) - chance) / max(1 - chance, 1e-9), 0, 1))


def measure(x):
    """Measure one real app (whole days of requests per minute)."""
    days = len(x) // 1440
    events, base, hot = find_bursts(x)
    calm = ~hot
    by_day = x[:days * 1440].reshape(days, 1440)
    padded = np.concatenate([by_day[:, -30:], by_day, by_day[:, :30]], axis=1)
    shapes = sliding_window_view(padded, 60, axis=1).mean(axis=2)[:, :1440] / x.mean()  # hourly average, wraps at midnight
    starts = events[:, 0].astype(int)
    extra = (events[:, 1] - 1) * base[starts] / x.mean()  # requests a burst adds on top, in units of the app's mean
    return {
        "shapes": shapes,                                        # slow daily shape of each real day
        "noise": float(np.std(x[calm] / base[calm] - 1)),         # spread of calm minutes
        "rate": len(events) / days,                               # bursts per day
        "events": np.column_stack([extra, events[:, 2:]]),        # extra load, length, minutes to peak
        "repeat": repeat_share(starts, days, np.random.default_rng(0)),
    }


def _add_burst(extra, t0, size, length, ramp):
    """Bursts add requests on top of normal traffic (for example a scheduled job calling the app)."""
    for i in range(int(length)):
        if 0 <= t0 + i < len(extra):
            extra[t0 + i] = max(extra[t0 + i], size * min(1.0, (i + 1) / (ramp + 1)))


class FittedTraffic:
    """Builds two-day traces (yesterday + today) from the measured statistics of real training apps.

    Each episode copies one real app: its slow daily shape (two consecutive real days, smoothed),
    its noise level, how many bursts it has per day, the height, length and ramp of each burst
    (resampled from its real bursts), and how often a burst repeats at the same time the next day.
    The noise and the bursts themselves are new, so every episode is a different pair of days.
    """

    def __init__(self, series_by_app):
        self.apps = [measure(x) for x in series_by_app]

    def __call__(self, rng):
        app = self.apps[rng.integers(len(self.apps))]
        k = rng.integers(len(app["shapes"]) - 1)
        load = np.concatenate(app["shapes"][k:k + 2]) * (1 + app["noise"] * rng.standard_normal(2880))
        extra = np.zeros(2880)
        first = app["events"][rng.integers(len(app["events"]), size=rng.poisson(app["rate"]))] if len(app["events"]) else []
        starts = rng.integers(0, 1440, len(first))
        for t0, ev in zip(starts, first):
            _add_burst(extra, t0, *ev)
            if rng.random() < app["repeat"]:  # the same burst again tomorrow, a few minutes off
                _add_burst(extra, t0 + 1440 + rng.integers(-3, 4), *ev)
        if len(app["events"]):
            for ev in app["events"][rng.integers(len(app["events"]), size=rng.poisson(app["rate"] * (1 - app["repeat"])))]:
                _add_burst(extra, rng.integers(1440, 2880), *ev)
        load = np.clip(load + extra, 0.0, None)
        return load * P99_TARGET / max(np.percentile(load, 99), 1e-9)
