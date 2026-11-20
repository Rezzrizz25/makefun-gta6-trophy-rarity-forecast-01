# makefun-gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Rezzrizz25/makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Rezzrizz25/makefun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast every Grand Theft Auto VI trophy's earn rate on Exophase at
2026-11-26 00:00 UTC — a week after launch — as five quantiles per trophy,
for make.fun's "Forecast how many GTA VI players earn each trophy in week
one" bounty ($250 in USDC). Built by an AI agent for make.fun's bounty.

The graded entry is `main.py` (Python 3.13, standard library only, offline,
deterministic). `index.html` is an interactive page showing the model's
methodology and an illustrative week-one forecast table.

## Model

A log-time trend in ceiling-logit space, with an anchor prior. See the header
comment in `main.py` for the full reasoning and numbers.

- `x = ln(1 + hours since 2026-11-19T00:00Z)`, `u = logit(rate / ceiling)`
- Per trophy: a Theil-Sen line of `u` on `x` over the last 24 h of reads
  (the median pairwise slope — a level jump in the feed cannot bend it the
  way it bends OLS), slope shrunk toward a prior slope, read at the target
  from the last observed level.
- The prior slope is the line through two real Exophase snapshots of the
  game's PS5 trophy page: the kit's launch-night history (638 players
  tracked) and the live page read 2026-11-20 18:51 UTC (Chapter 1 at
  11.98%, Beat Cal fishing at 13.30%, most late-story and grind trophies
  diluted to the 0.01% print floor). That dilution event — the tracked base
  growing ~100x while earners accumulate slowly — dominates early
  trophy-rate dynamics, and the launch-night history is flat, so a
  history-only fit would just say "persistence".
- Ceilings per trophy from tier plus what the trophy asks for: the Platinum
  sits at 0.02% (100% completion needs ~200 h of play), story chapters climb
  furthest, the first mission is already at ~100%, grinds barely move in
  week one.
- Spread: a random-walk term, the fit's residual, slope uncertainty, and a
  prior-uncertainty term when the forecast leans on the prior. Floor
  trophies (last rate at the 0.01% print floor) get a wide band — the print
  is censored.

## Backtests

From the kit's practice cases and synthetic histories built on the two real
anchors (20 seeds each; mean pinball loss on the logit scale, lower better):

| check | this model | persistence | house trend |
|---|---|---|---|
| synthetic 2-day histories, known day-7 truth | **0.129** | 0.227 | 0.219 |
| same, with per-read noise + persistent level jumps | **0.208** | 0.238 | 0.273 |
| anchor-2-as-answer (consistency check) | 1.88 | 2.87 | 2.87 |

Practice cases p00/p01: valid output, deterministic across runs, ~0.05 s.
Edge cases (null rates, one read, 0%/100% rates, unknown trophy id) all
produce valid, non-decreasing, finite rows.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

One `<id>.in` in → one `<id>.out` out:

```json
{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}
```

## Sources

- Trophy rates: [Exophase — Grand Theft Auto VI Trophies, PS5](https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/)
- Launch sales: Sensor Tower estimates via Notebookcheck/VGC — ~24.6 million
  copies, ~$2.36 billion on day one (19 November 2026); ~75% on PlayStation 5.
- Trophy list: 55 trophies — 1 Platinum, 2 Gold, 6 Silver, 46 Bronze;
  Platinum needs 100% completion; ~200 h estimated for everything (VGC).
- Competition: [make.fun — gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
