# rl-autoscaler

Can a reinforcement learning (RL) policy keep fewer requests waiting than the Kubernetes autoscaler, at the same cost, by learning when to keep spare servers? Humanitarians AI RL Environments project.

## Problem

Threshold autoscalers, such as the Kubernetes Horizontal Pod Autoscaler (HPA), react to load after it arrives, and new servers take minutes to boot. In real traffic most bursts jump in a single minute, so no policy can react to them in time: the only defence is spare capacity. But many bursts repeat at the same time each day, so a policy can learn when spare capacity pays off.

## Data

Real traffic comes from the [Azure Functions Trace 2019](https://github.com/Azure/AzurePublicDataset) (Shahrad et al., USENIX ATC 2020), licensed [CC-BY 4.0](https://github.com/Azure/AzurePublicDataset/blob/master/LICENSE). [prepare_azure.py](scripts/prepare_azure.py) sums the HTTP-triggered functions of each app per minute and keeps 19 busy apps that vary (rule in [the metadata](data/azure2019_http_apps.json)); ids are shortened. 14 apps are for training (days 1-10); unseen days 11-14 and 5 held-out apps are for testing.

Measured on the training days: per-minute noise around 15%, about 24 bursts per app per day, and 72% of bursts peak in their first minute. The synthetic generator ([traffic.py](src/rl_autoscaler/traffic.py)) copies these statistics from one real app per episode (daily shape, noise, burst sizes, lengths, ramps, and how often a burst repeats the next day) and draws new noise and bursts.

## Approach

- `AutoscaleEnv` ([env.py](src/rl_autoscaler/env.py)) is a Gymnasium environment. One step is one minute. The agent adds or removes up to 5 servers (600 requests per minute each); new servers serve after a 3-minute boot delay but are billed at once. Requests not served in the minute they arrive are late; past 2 minutes of queue they are dropped. Each episode is two days: the agent acts on the second and can see the first.
- Observation: the last 10 minutes of load, yesterday's load for the next 10 minutes, running and booting servers, queue size, time of day.
- Reward: minus server cost, minus a penalty per late request.
- Baselines ([policies.py](src/rl_autoscaler/policies.py)): a fixed count sized for the day's peak; the HPA rule (70% target, 10% tolerance, 5-minute scale-down window); the same rule on a linear-trend forecast; the same rule on a "same time yesterday" forecast.

## Results so far

Five held-out apps, 13 test days each (`python scripts/demo.py`):

| Policy | Server-minutes per day | Late requests |
|---|---|---|
| Fixed, sized for the day's peak | 39,445 | 0.07% |
| Threshold (Kubernetes HPA) | 19,986 | 5.47% |
| Linear forecast | 22,858 | 4.63% |
| Same time yesterday | 24,488 | 1.17% |

![A held-out day: capacity under the threshold rule and under "same time yesterday"](results/demo.png)

Spare servers alone do not explain the gap: the threshold rule at a 50% target costs more (27,371 server-minutes per day) and is still late 3.10% of the time. Daily repetition is the signal to learn, so the RL policy has to beat "same time yesterday" at equal cost.

### PPO (week of 9 Nov)

Plain PPO, and residual PPO that adjusts the "same time yesterday" action by -5 to +5 servers, each at 5 late penalties with 3 seeds, trained on Modal ([train.py](src/rl_autoscaler/train.py), [modal_train.py](scripts/modal_train.py)):

![Cost against late requests for the baselines and the PPO agents](results/frontier.png)

No PPO agent beats "same time yesterday" at equal cost yet; the best, residual PPO at the highest penalty, only ties it.

### Weekly pattern

The 14 days have a weekly shape: days 6-7 and 13-14 are the quietest, like weekends, and day 8, the day after the first pair, is the busiest. On that day "same time yesterday" plans from a quiet day. 14 days give each app only one such day, too few to test how a policy handles it.

## Run

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,train]'
pytest
python scripts/demo.py
python scripts/frontier.py
```

To train the PPO agents again: `modal setup`, then `modal run --detach scripts/modal_train.py`, then `modal volume get rl-autoscaler-results ppo results/ --force`. The result files are in `results/ppo/`; the trained models stay on the Modal volume.

The processed data is in `data/`. To rebuild it, download the trace into `data/raw/azure2019`, `pip install -e '.[data]'`, and run `python scripts/prepare_azure.py`.

## Plan

- Week of 2 Nov: environment, baselines, real traffic, fitted synthetic traffic, tests (done).
- Week of 9 Nov: PPO with Stable-Baselines3 on Modal, trained on fitted synthetic and real training days: plain PPO, and residual PPO that corrects "same time yesterday" (done, results above).
- Week of 16 Nov: real traces several weeks long with weekday and weekend patterns, so every test includes days after a weekend; a "same time last week" baseline; test the baselines on them first.
- Week of 23 Nov: PPO on the new traces, with a validation split for choosing settings; final tests, report, `SKILL.md`.
- If PPO beats the rule: the same method on KV-cache eviction or quantization choice in LLM serving.
