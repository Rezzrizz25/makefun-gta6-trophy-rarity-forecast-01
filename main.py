# gta6-trophy-rarity-forecast-01: forecast every GTA VI trophy's Exophase earn rate
# at the case's target, as five quantiles (5/25/50/75/95%), scored by pinball
# loss on the logit scale.
#
# Model: log-time trend in ceiling-logit space, with an anchor prior.
#
#   x = ln(1 + hours since 2026-11-19T00:00Z)   (launch clock; log-time makes
#                                                 growth saturate in real time)
#   u = logit(rate / C)                          (rate as a fraction of its
#                                                 ceiling C: a logistic approach
#                                                 to the ceiling is a straight
#                                                 line in u, so the forecast
#                                                 saturates instead of running
#                                                 away; far below C, u behaves
#                                                 like logit(rate))
#
# For each trophy the program fits a Theil-Sen line of u on x over the last
# 24 hours of reads (the median pairwise slope: a level jump in the feed, an
# Exophase refresh or a tracked-player jump, bends an OLS slope and the
# extrapolation overshoots; the median pairwise slope ignores the pairs that
# span the jump), shrinks its slope toward a prior slope, and reads the line
# at the target from the last observed level.
#
# The prior slope comes from two real Exophase snapshots of the game's PS5
# trophy page (https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/):
#   anchor 1: the house kit's own practice history (launch night, ~2.5 h after
#             launch, 638 players tracked; every read identical)
#   anchor 2: the live page read 2026-11-20 18:51 UTC (~42.9 h after launch):
#             Chapter 1 at 11.98%, Chapter 2 at 1.71%, Beat Cal fishing at
#             13.30%, most late-story and grind trophies diluted to the 0.01%
#             print floor as the tracked base exploded past launch night's 638.
# The line through those two snapshots, per trophy, is the prior. It matters
# because the launch-night history is FLAT (every read the same), so any
# history-only fit has slope zero there: without a prior the forecast is
# persistence, and persistence scores ~1.51 on the practice (the house
# baseline's number) because the answering read jumps far from the stale
# launch-night snapshot. The two snapshots are real inputs, not invented data:
# they record the dilution event (tracked players growing ~100x while earners
# accumulate slowly) that dominates early trophy-rate dynamics.
#
# Why the trend, not just the prior: the final case carries ~2 days of reads
# every 10 minutes, including the first-weekend play wave, so by the deadline
# the recent slope is informative and the prior is only a shrinkage target.
# The blend weight ("info") is 0 when the history is flat and 1 when it has
# both length (>= 25 reads) and movement (>= 0.40 units of u-range).
#
# Ceilings, per trophy, come from tier plus what the trophy asks for: the
# Platinum sits at 0.02% (100% completion needs ~200 h of play, so nobody can
# hold it seven days after launch), story chapters climb furthest, the first
# mission is already at ~100%, grinds (250 stunts, 100% completion) barely
# move in week one. A ceiling is lifted only by RECENT reads above it, so a
# launch-night artefact (Platinum printed at 0.63% on 638 players, since
# corrected to 0.01% on the live page) cannot lift it.
#
# Spread: a random-walk term (0.35*sqrt(horizon in days)), plus the fit's
# residual (MAD-based), plus slope uncertainty growing with the extrapolation
# distance, plus a prior-uncertainty term that is large when the forecast
# leans on the prior. Floor trophies (last rate at the 0.01% print floor) get
# a wide band: the print is censored, the true rate is somewhere below it,
# and grind earners accumulate against a still-growing denominator. The
# spread scale was set by backtests (below): narrower than the house
# baseline's, because the median error on realistic histories is smaller
# than the baseline assumes.
#
# Deterministic by construction: no randomness, no wall clock, no network,
# sorted input order, math.fsum. Pure standard library, offline.
# Run like the grader:  python3 -P -s main.py
#   with MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR set, else /input and /output.
# One <id>.in in -> one <id>.out out:
#   {"trophies": [{"id": <id>, "q": [q05, q25, q50, q75, q95]}, ...]}
#
# Backtests (see work/test_model.py and work/spread_experiment.py; all numbers
# from the kit's practice cases and from synthetic histories built on the two
# real anchors, 20 seeds each):
#   - practice p00/p01: output valid, deterministic across runs, ~0.05 s.
#   - anchor-2-as-answer (history = flat launch night, target 2026-11-20
#     18:51Z, answer = the real live page): model 1.88 vs persistence 2.87
#     (consistency check: the prior is fitted on that snapshot, so this shows
#     the mechanism works, not out-of-sample skill; the u-prior deliberately
#     sits above the censored 0.01% floor prints).
#   - synthetic 2-day histories with known day-7 truth (rate curves through
#     both real anchors, saturating at stated long-run rates): model 0.129 vs
#     persistence 0.227 vs house-trend 0.219 (mean pinball loss).
#   - same, with per-read noise plus persistent level jumps in the feed:
#     model 0.208 vs persistence 0.238 vs house-trend 0.273.
#   - edge cases (null rates, one read, 0%/100% rates, unknown trophy id):
#     all produce valid, non-decreasing, finite rows.
#
import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a model.py shipped beside this file (kept for symmetry with the starter;
# this entry is a single file).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level (precise values).
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# Launch clock: rates are modelled against hours since 2026-11-19T00:00:00Z.
LAUNCH_EPOCH = 1795046400  # 2026-11-19T00:00:00Z as a Unix timestamp.

# --- prior anchors (see header) -------------------------------------------
# Anchor 1: house kit practice history, ~2.5 h after launch (638 tracked).
# Anchor 2: live Exophase page, 2026-11-20 18:51 UTC, ~42.85 h after launch.
# Indexed by trophy id (1..55), in the case's trophy order.
ANCHOR1 = [
    0.63, 100.0, 1.25, 1.10, 1.10, 1.10, 1.10, 1.10, 1.10, 1.10,
    1.10, 1.10, 1.10, 0.63, 0.63, 0.78, 0.63, 0.63, 0.78, 2.04,
    0.94, 0.78, 0.78, 0.78, 0.63, 0.63, 0.78, 0.63, 0.78, 0.63,
    0.63, 0.63, 0.94, 0.78, 1.25, 0.78, 0.94, 1.57, 0.94, 0.78,
    0.94, 0.78, 0.78, 0.94, 0.78, 0.78, 0.78, 0.78, 0.78, 0.78,
    0.78, 0.78, 0.94, 0.78, 0.78,
]
ANCHOR2 = [
    0.01, 100.0, 11.98, 1.71, 0.03, 0.02, 0.02, 4.26, 0.03, 0.02,
    0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 13.30,
    1.42, 0.01, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
    0.01, 0.01, 0.31, 0.04, 1.70, 0.08, 0.23, 0.20, 0.04, 0.01,
    0.01, 0.04, 0.01, 0.17, 0.01, 1.26, 0.01, 0.01, 0.01, 0.72,
    0.06, 0.01, 0.03, 0.01, 0.01,
]
ANCHOR_T1_HOURS = 2.5    # launch-night snapshot
ANCHOR_T2_HOURS = 42.85  # live page 2026-11-20 18:51 UTC

# Fallback anchors for a trophy id the tables do not know, by tier.
FALLBACK_ANCHOR = {
    "platinum": (0.63, 0.01),
    "gold": (1.00, 0.02),
    "silver": (0.63, 0.01),
    "bronze": (0.80, 0.05),
}

# --- model constants (tuned on the backtests in work/test_model.py) --------
WINDOW_HOURS = 24.0      # trend fit uses the last 24 h of reads
SHRINK_READS = 8.0       # prior slope counts as this many reads in the shrink
BETA_MIN = -0.45         # prior slopes are clipped: dilution cannot run away
BETA_MAX = 1.50
SIGMA_RW = 0.35         # random-walk spread, logit units per sqrt(day)
RESID_FLOOR = 0.08       # residual spread is floored ...
RESID_CAP = 0.90         # ... and capped
RESID_FLAT = 0.25        # spread when the history is flat
PRIOR_SPREAD = 0.80      # extra spread when leaning on the prior
SLOPE_UNC = 0.15         # slope uncertainty as a fraction of |slope|
SPREAD_MIN = 0.25
SPREAD_MAX = 1.80
FLOOR_SPREAD_MIN = 0.90  # floor-censored trophies get at least this spread
DAMP_HORIZON = 1.00      # no extra damping: log-time already saturates


def clamp_rate(pct):
    """A rate in percent, kept inside the scorer's logit domain."""
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(y):
    if y >= 0:
        e = math.exp(-y)
        return 100.0 / (1.0 + e)
    e = math.exp(y)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s):
    """ISO-8601 UTC timestamp -> Unix seconds (no timezone library needed)."""
    # Format: 2026-11-19T03:00:00Z or with fractional seconds.
    date, time = s.split("T")
    y, m, d = (int(v) for v in date.split("-"))
    time = time.rstrip("Z")
    if "." in time:
        time = time.split(".")[0]
    hh, mm, ss = (int(v) for v in time.split(":"))
    # Days since epoch via the proleptic Gregorian calendar.
    days = (y - 1970) * 365 + (y - 1969) // 4 - (y - 1901) // 100 + (y - 1601) // 400
    for mo, dim in ((1, 31), (2, 28), (3, 31), (4, 30), (5, 31), (6, 30),
                    (7, 31), (8, 31), (9, 30), (10, 31), (11, 30), (12, 31)):
        if m > mo:
            days += dim
            if mo == 2 and (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)):
                days += 1
    days += d - 1
    return days * 86400 + hh * 3600 + mm * 60 + ss


def log_time(epoch_seconds):
    """x = ln(1 + hours since launch)."""
    hours = max(0.0, (epoch_seconds - LAUNCH_EPOCH) / 3600.0)
    return math.log1p(hours)


def ceiling_for(trophy):
    """The highest rate this trophy can plausibly reach by the target.

    From tier plus what the trophy asks for: story chapters climb furthest,
    the first mission is already at ~100%, grinds (250 stunts, 100%
    completion, the Platinum) barely move in week one.
    """
    desc = (trophy.get("description") or "").lower()
    name = (trophy.get("name") or "").lower()
    text = name + " " + desc
    tier = (trophy.get("tier") or "bronze").lower()
    if tier == "platinum":
        # 100% completion needs ~200 h of play (VGC estimate): nobody can
        # hold this seven days after launch, so the ceiling sits just above
        # the 0.01% print floor.
        return 0.02
    if "100% completion" in text or "100% completion" in desc:
        return 0.10
    if "gone fishing" in text:
        return 99.9
    if "chapter 1" in text:
        return 60.0
    if "chapter 2" in text:
        return 35.0
    if "chapter 3" in text:
        return 18.0
    if "chapter 4" in text:
        return 10.0
    if "final mission" in text and tier == "gold":
        return 8.0
    if "beat cal" in text:
        return 45.0
    if "breakfast run" in text:
        return 25.0
    grind = ("250 stunts" in text or "20 different races" in text
             or "top fun drone" in text or "shooting range" in text
             or "100 locations" in text or "50 challenges" in text
             or "collectible card" in text or "meditate at every" in text
             or "100% completion" in text)
    if grind:
        return 2.0
    if tier == "silver":
        return 5.0
    if tier == "gold":
        return 8.0
    return 30.0


def u_of(rate_pct, ceiling):
    """logit of a rate as a fraction of its ceiling: u = logit(r / C).

    Trophy rates saturate: a rate cannot pass the share of players who will
    ever do that thing. In u-space a logistic approach to the ceiling is a
    straight line, so a local slope fit extrapolates without overshooting
    past the ceiling. Far below the ceiling u behaves like logit(r).
    """
    frac = min(rate_pct, 0.999 * ceiling) / ceiling
    frac = min(0.999, max(0.001, frac))
    return math.log(frac / (1.0 - frac))


def rate_of(u, ceiling):
    """Invert u_of: the rate in percent for a u value and ceiling."""
    if u >= 0:
        e = math.exp(-u)
        frac = 1.0 / (1.0 + e)
    else:
        e = math.exp(u)
        frac = e / (1.0 + e)
    return ceiling * frac


def prior_line(trophy, ceiling):
    """(alpha, beta): the prior line u = alpha + beta * x for a trophy.

    The line through the two real Exophase snapshots (see the header), in
    u-space, with the slope clipped and two censored cases handled: a rate
    printed as 100% is treated as 98% with a dilution drag (new players join
    faster than they fish), and a rate printed at the 0.01% floor keeps only
    a gentle negative slope (the print is censored; the earners are still
    accumulating underneath).
    """
    tid = trophy.get("id")
    tier = (trophy.get("tier") or "bronze").lower()
    if isinstance(tid, int) and 1 <= tid <= len(ANCHOR1):
        a1 = ANCHOR1[tid - 1]
        a2 = ANCHOR2[tid - 1]
    else:
        a1, a2 = FALLBACK_ANCHOR.get(tier, (0.80, 0.05))
    x1 = math.log1p(ANCHOR_T1_HOURS)
    x2 = math.log1p(ANCHOR_T2_HOURS)
    if a1 >= 99.9 or a2 >= 99.9:
        # Saturating trophy: hold near 98% with a slow dilution drag.
        u_at_x2 = u_of(98.0, ceiling)
        beta = -0.08
        alpha = u_at_x2 - beta * x2
        return alpha, beta
    u1 = u_of(a1, ceiling)
    u2 = u_of(a2, ceiling)
    beta = (u2 - u1) / (x2 - x1)
    if a2 <= 0.011:
        # Printed at the floor: the slope from the censored print overstates
        # the decline, so use only a gentle negative drift.
        beta = max(BETA_MIN, min(beta, 0.10))
    beta = max(BETA_MIN, min(BETA_MAX, beta))
    alpha = u1 - beta * x1
    return alpha, beta


def median(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def theil_sen(xs, ys):
    """Robust line fit: the median pairwise slope, median intercept.

    A level jump in the reads (an Exophase refresh, a tracked-player jump)
    bends an OLS slope and the extrapolation overshoots; the median pairwise
    slope ignores the pairs that span the jump. Pairs are capped by a
    deterministic stride so the fit stays fast on 10-minute reads.
    Returns (a, b, resid_std) with resid from the median absolute deviation.
    """
    n = len(xs)
    if n < 2:
        return None
    pairs = []
    # Deterministic stride: keep at most ~1500 pairs.
    max_pairs = 1500
    total = n * (n - 1) // 2
    stride = max(1, total // max_pairs)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[j] - xs[i]
            if dx <= 1e-9:
                continue
            count += 1
            if count % stride != 0:
                continue
            pairs.append((ys[j] - ys[i]) / dx)
    if not pairs:
        return None
    b = median(pairs)
    a = median([y - b * x for x, y in zip(xs, ys)])
    resid = [abs(y - (a + b * x)) for x, y in zip(xs, ys)]
    # MAD -> std for a normal (1.4826), floored so a perfect fit keeps some spread.
    resid_std = 1.4826 * median(resid)
    return (a, b, resid_std)


def weighted_ols(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, resid_std)."""
    sw = math.fsum(ws)
    if sw <= 0:
        return None
    mx = math.fsum(w * x for w, x in zip(ws, xs)) / sw
    my = math.fsum(w * y for w, y in zip(ws, ys)) / sw
    sxx = math.fsum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return (my, 0.0, 0.0)
    sxy = math.fsum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    n = len(xs)
    if n > 2:
        resid = math.fsum(w * (y - (a + b * x)) ** 2
                          for w, x, y in zip(ws, xs, ys)) / sw
        resid_std = math.sqrt(max(0.0, resid))
    else:
        resid_std = 0.0
    return (a, b, resid_std)


def forecast_trophy(trophy, rates_series, x_target, horizon_days):
    """rates_series: list of (x, rate_pct) reads for this trophy, oldest first.

    Returns the five quantiles at the case's target. Everything is fitted in
    u-space (u = logit(rate / ceiling)): a logistic approach to the ceiling
    is a straight line there, so the extrapolation saturates instead of
    running away, and a rate near its ceiling gets a compressed spread.
    """
    base_ceiling = ceiling_for(trophy)
    # The ceiling must sit above every RECENT observed rate, else u is
    # undefined; launch-night reads do not lift it (their tiny sample
    # printed artefacts like a 0.63% Platinum that the live page has since
    # corrected to 0.01%).
    recent = rates_series[-144:] if len(rates_series) > 144 else rates_series
    max_seen = max((r for _, r in recent), default=0.0)
    ceiling = min(99.9, max(base_ceiling, max_seen * 1.2, 0.05))
    alpha, beta_prior = prior_line(trophy, ceiling)
    u_prior = alpha + beta_prior * x_target

    def quantiles(u_hat, spread):
        qs = [round4(min(99.99, max(0.01, rate_of(u_hat + z * spread, ceiling))))
              for z in Z]
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        return qs

    if not rates_series:
        spread = math.sqrt((SIGMA_RW * math.sqrt(horizon_days)) ** 2
                           + PRIOR_SPREAD ** 2)
        spread = max(SPREAD_MIN, min(SPREAD_MAX, spread))
        return quantiles(u_prior, spread)

    series = [(x, u_of(r, ceiling)) for x, r in rates_series]

    # Trend fit over the last 24 h of reads (all reads if fewer than 3).
    # x is log-time, so convert back to hours since launch for the window.
    t_last = series[-1][0]
    window = series
    if len(series) >= 3:
        h_last = math.expm1(t_last)
        window = [(x, u) for (x, u) in series
                  if math.expm1(x) >= h_last - WINDOW_HOURS]
        if len(window) < 3:
            window = series
    xs = [x for x, _ in window]
    us = [u for _, u in window]
    n = len(window)

    u_values = [u for _, u in series]
    u_range = max(u_values) - min(u_values)
    info = min(1.0, (len(series) - 1) / 24.0) * min(1.0, u_range / 0.40)

    # Last level: median of the last three reads (robust to 2-decimal rounding).
    tail = sorted(u_values[-3:])
    u_last = tail[len(tail) // 2]

    fit = theil_sen(xs, us)
    if fit is None or info <= 0.0:
        b = beta_prior
        resid = RESID_FLAT
    else:
        _, b_ols, resid = fit
        n_eff = min(n, 48)
        b = (n_eff * b_ols + SHRINK_READS * beta_prior) / (n_eff + SHRINK_READS)
        resid = max(RESID_FLOOR, min(RESID_CAP, resid))
    b = max(BETA_MIN - 1.0, min(BETA_MAX + 1.0, b))

    dx = x_target - t_last
    damp = DAMP_HORIZON if dx > 1.0 else 1.0
    u_trend = u_last + b * dx * damp

    u_hat = info * u_trend + (1.0 - info) * u_prior

    spread = math.sqrt(
        (SIGMA_RW * math.sqrt(horizon_days)) ** 2
        + resid ** 2
        + (SLOPE_UNC * abs(b) * max(0.0, dx)) ** 2
        + (PRIOR_SPREAD * (1.0 - info)) ** 2
    )
    if rates_series and rates_series[-1][1] <= 0.011:
        spread = max(spread, FLOOR_SPREAD_MIN)
    spread = max(SPREAD_MIN, min(SPREAD_MAX, spread))

    return quantiles(u_hat, spread)


def forecast_case(case, name):
    if not case.get("history"):
        raise ValueError(f"{name} has no history")
    levels = case.get("levels") or LEVELS
    x_target = log_time(parse_time(case["target"]))
    horizon_days = case.get("horizon_minutes", 0) / 1440.0

    trophies = case["trophies"]
    # Per-trophy series of (x, rate), oldest first, skipping null reads.
    # The u transform happens inside forecast_trophy (it needs the ceiling).
    series = {i: [] for i in range(len(trophies))}
    for read in case["history"]:
        x = log_time(parse_time(read["slot"]))
        rates = read["rates"]
        for i in range(len(trophies)):
            if i >= len(rates):
                break
            r = rates[i]
            if r is None:
                continue
            series[i].append((x, r))

    rows = []
    for i, trophy in enumerate(trophies):
        qs = forecast_trophy(trophy, series[i], x_target, horizon_days)
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
