#!/usr/bin/env python3
"""Calibrated forecaster for pumpfun-top5-forecast-01.

Order: honest per-coin beliefs per place (rank kernel x momentum x venue x age),
total known mass left at a horizon-dependent level, deep chart members named
at p=0 so unnamed places (coins born after the cut) share a thicker mass.
Sizes: a per-place mixture over the cap LEVEL each place will sit at, not any
one coin's cap: incumbent persistence, fresh graduate pump, champion, crash,
and a uniform safety floor, widened by horizon.
"""

import json
import math
import os
import random
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
MICRO = 1_000_000
BASE_FLOOR = 0.02
MC_SEED = 9000
SQRT2 = math.sqrt(2.0)


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def erfc(x):
    """Numerical Recipes' complementary error function approximation."""
    z = abs(x)
    t = 1.0 / (1.0 + 0.5 * z)
    r = t * math.exp(
        -z * z - 1.26551223
        + t * (1.00002368 + t * (0.37409196 + t * (0.09678418
        + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398
        + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))))
    )
    return r if x >= 0 else 2.0 - r


def norm_mass(lo, hi, mean, sd):
    """P(lo <= X < hi) for X ~ N(mean, sd) in log10 space."""
    if hi <= lo:
        return 0.0
    plo = 0.0 if lo == -math.inf else 0.5 * erfc((lo - mean) / (sd * SQRT2))
    phi = 0.0 if hi == math.inf else 0.5 * erfc((hi - mean) / (sd * SQRT2))
    return max(0.0, plo - phi)


def mixture_bins(place, edges, components, uniform_weight, uniform_lo, uniform_hi):
    """components: [(weight, center, sd)] on log10 cap; plus a uniform floor."""
    raw = []
    for b in range(len(edges) + 1):
        lo = -math.inf if b == 0 else edges[b - 1]
        hi = math.inf if b == len(edges) else edges[b]
        p = BASE_FLOOR / (len(edges) + 1)
        for weight, center, sd in components:
            p += weight * norm_mass(lo, hi, center, sd)
        overlap_lo = max(lo, uniform_lo)
        overlap_hi = min(hi, uniform_hi)
        if overlap_hi > overlap_lo and uniform_hi > uniform_lo:
            p += uniform_weight * (overlap_hi - overlap_lo) / (uniform_hi - uniform_lo)
        raw.append(p)
    total = sum(raw)
    if total <= 0.0:
        raw = [1.0] * len(raw)
        total = float(len(raw))
    units = [int(MICRO * x / total) for x in raw]
    left = MICRO - sum(units)
    order = sorted(range(len(raw)), key=lambda b: (raw[b] / total * MICRO - units[b], -b), reverse=True)
    for b in order[:left]:
        units[b] += 1
    return [{"place": place, "bin": b, "p": units[b] / MICRO} for b in range(len(units)) if units[b]]


def mc_levels(place, logcaps, venues, h, draws=400):
    """Monte Carlo cap LEVELS at a place: survivors drift and die, stars are
    born around the graduation band. Deterministic per place."""
    rng = random.Random(MC_SEED + place)
    rate = 12.0 * h
    limit = math.exp(-rate)
    p = 1.0
    n_star = 0
    while True:
        p *= rng.random()
        if p <= limit:
            break
        n_star += 1
    conv = min(0.12 * (h ** 0.6), 0.85)
    out = []
    for _ in range(draws):
        vals = []
        for lg, ven in zip(logcaps, venues):
            death = 4.5 if ven != "pool" else 0.9
            if rng.random() < 1.0 - math.exp(-death * h):
                continue
            sig = min((0.28 if ven != "pool" else 0.85) * math.sqrt(max(h, 1e-3)), 2.0)
            vals.append(lg + 0.012 * h - conv * (lg - 3.30) + rng.gauss(0, sig))
        for _ in range(n_star):
            vals.append(rng.gauss(3.28, 0.22))
        vals.sort(reverse=True)
        out.append(vals[place - 1] if len(vals) >= place else 1.5)
    return out


def mc_bins(place, edges, logcaps, venues, h, alpha):
    """Smoothed MC histogram per place, mixed with a uniform base floor."""
    n = len(edges) + 1
    hist = [0.0] * n
    for lv in mc_levels(place, logcaps, venues, h):
        b = 0 if lv < edges[0] else (n - 1 if lv >= edges[-1] else int((lv - edges[0]) / 0.05) + 1)
        for d, w in ((0, 0.25), (1, 0.5), (-1, 0.25)):
            bb = b + d
            if 0 <= bb < n:
                hist[bb] += w
    total = sum(hist) or 1.0
    return [alpha * x / total + (1.0 - alpha) * BASE_FLOOR / n for x in hist]


def main():
    task = json.loads((INPUT_DIR / "task.json").read_text())
    marks = []
    for line in (INPUT_DIR / "history.ndjson").read_text().splitlines():
        if not line:
            continue
        mark = json.loads(line)
        if mark.get("ok") and isinstance(mark.get("rows"), list) and mark["rows"]:
            for row in mark["rows"]:
                row["rank"] = int(row["rank"])
                row["cap"] = float(row["market_cap_lamports"])
            mark["rows"].sort(key=lambda row: row["rank"])
            marks.append(mark)
    if not marks:
        raise RuntimeError("history has no usable priced marks")
    places = int(task["places"])
    if len(marks[-1]["rows"]) < places:
        raise RuntimeError("last mark has fewer rows than places")

    last = marks[-1]
    chart = last["rows"]
    t0 = timestamp(task["t0"])
    cut = timestamp(last["at"])
    reads = [timestamp(t) for t in task.get("reads", [])]
    h = max(0.0, (sum(reads) / len(reads) - cut) / 3600.0) if reads else 1.0

    # --- per-coin features -------------------------------------------------
    # momentum over the last hour and stability over the whole history beat
    # a single 10-minute delta, which is mostly noise.
    window = marks[-7:]
    cols = {}
    for mark in window:
        for row in mark["rows"]:
            slot = cols.setdefault(row["mint"], {"caps": [], "ranks": []})
            slot["caps"].append(row["cap"])
            slot["ranks"].append(row["rank"])
    span = max(1e-9, (cut - timestamp(window[0]["at"])) / 3600.0)
    appearances = {}
    for mark in marks:
        for row in mark["rows"]:
            appearances[row["mint"]] = appearances.get(row["mint"], 0) + 1

    candidates = {}
    union = {row["mint"] for row in chart}
    for mark in window[-2:]:
        union |= {row["mint"] for row in sorted(mark["rows"], key=lambda r: r["rank"])[:120]}
    for mint in union:
        row = next((r for r in chart if r["mint"] == mint), None)
        if row is None:
            for mark in reversed(window):
                row = next((r for r in mark["rows"] if r["mint"] == mint), None)
                if row is not None:
                    break
            if row is None:
                continue
            row = dict(row)
            row["rank"] = 200
        slot = cols.get(mint, {"caps": [row["cap"]], "ranks": [row["rank"]]})
        cnow = row["cap"]
        vlog = math.log10(max(slot["caps"][-1], 1.0)) - math.log10(max(slot["caps"][0], 1.0))
        vhat = max(-2.0, min(2.5, vlog / span))
        stability = sum(1 for r in slot["ranks"] if r <= 25) / float(len(slot["ranks"]))
        age_h = max(0.0, (cut - timestamp(row["created_at"])) / 3600.0)
        candidates[mint] = {
            "rank": row["rank"],
            "cap": cnow,
            "vhat": vhat,
            "age": age_h,
            "venue": row.get("venue", "curve"),
            "seen": appearances.get(mint, 1),
            "stab": stability,
        }

    # known-coin mass per place: short horizons keep more incumbents, long
    # horizons hand the places to coins born after the cut.
    s_known = 0.52 - 0.032 * max(0.0, min(h - 1.5, 9.0))
    s_known = max(0.20, min(0.55, s_known))
    lam = max(0.14, 0.22 - 0.009 * max(0.0, h - 1.5))
    # Naming a coin below the unnamed share it would otherwise receive is
    # self-defeating: name it well or leave it unnamed. Expected read chart
    # holds ~70-80 ids we cannot name (fresh launches), so the unnamed
    # share lands near (1-s_known)/78.
    q_min = max(0.0015, (1.0 - s_known) / 200.0)

    probabilities = []
    aggregate = {}
    rank_by_mint = {row["mint"]: row["rank"] for row in chart}
    for place in range(1, places + 1):
        weights = {}
        for mint, feat in candidates.items():
            w = math.exp(-lam * abs(feat["rank"] - place))
            w *= math.exp(0.45 * feat["vhat"])
            if feat["venue"] == "pool":
                w *= 1.20 + (0.40 + 0.03 * min(h, 10.0)) * feat["stab"]
            else:
                w *= 1.0 + (0.45 + 0.03 * min(h, 10.0)) * feat["stab"]
                if feat["age"] < 4.0:
                    w *= 1.25
            w *= 1.0 + 0.03 * min(feat["seen"], 12)
            weights[mint] = w
        total = sum(weights.values())
        named = []
        junk = []
        for mint, w in weights.items():
            q = s_known * w / total
            cap_sol = candidates[mint]["cap"] / 1e9
            if q >= q_min:
                named.append({"mint": mint, "place": place, "p": q})
            elif cap_sol < 900.0:
                # coins we would name too low to matter: named at p=0 they
                # still leave the unnamed pool when they sit on the read
                # chart, fattening the share the unnameable fresh launches
                # receive. Highest caps survive to the read chart, so those
                # go first.
                junk.append({"mint": mint, "place": place, "p": 0.0, "cap_sol": cap_sol})
        named.sort(key=lambda r: (-r["p"], r["mint"]))
        junk.sort(key=lambda r: (-r["cap_sol"], r["mint"]))
        rows = named + junk
        for r in rows:
            r.pop("cap_sol", None)
        named_sum = sum(r["p"] for r in rows)
        if named_sum > 1.0:
            scale = 1.0 / named_sum
            for r in rows:
                r["p"] *= scale
        rows = rows[:100]
        probabilities.extend(rows)
        for r in rows:
            if r["p"] > 0:
                aggregate[r["mint"]] = aggregate.get(r["mint"], 0.0) + r["p"]

    # --- sizes: the cap LEVEL each place sits at, as a mixture --------------
    edges = task["size"]["edges"]
    logcaps = [math.log10(max(chart[k - 1]["cap"], 1.0) / 1e9) for k in range(1, places + 1)]
    mean_level = sum(logcaps) / len(logcaps)
    hw = min(h, 7.0)
    # At long horizons the place levels drift away from the cut's levels and
    # the graduation band (~2k SOL) plus survivor growth take the weight.
    wA = 0.44 - 0.010 * hw
    wB = 0.17 + 0.010 * hw
    wC = 0.10 + 0.010 * hw
    wD = 0.13 - 0.006 * hw
    wE = max(0.09, 1.0 - wA - wB - wC - wD)
    sizes = []
    venues = [row.get("venue", "curve") for row in chart[:100]]
    logcols = [math.log10(max(row["cap"], 1.0) / 1e9) for row in chart[:100]]
    for k in range(1, places + 1):
        lk = 0.70 * logcaps[k - 1] + 0.30 * mean_level
        sdA = 0.20 + 0.022 * hw
        comp = [
            (wA, lk, sdA),
            (wB, 3.34, 0.22 + 0.008 * hw),
            (wC, mean_level + 0.40 + 0.020 * hw, 0.40 + 0.012 * hw),
            (wD, mean_level - 0.60, 0.50),
        ]
        mix = mc_bins(k, edges, logcols, venues, h, 0.5)
        raw = []
        n_bins = len(edges) + 1
        for b in range(n_bins):
            lo = -math.inf if b == 0 else edges[b - 1]
            hi = math.inf if b == len(edges) else edges[b]
            p = BASE_FLOOR / n_bins
            for weight, center, sd in comp:
                p += weight * norm_mass(lo, hi, center, sd)
            overlap_lo, overlap_hi = max(lo, 1.9), min(hi, 5.4)
            if overlap_hi > overlap_lo:
                p += max(0.0, 1.0 - wA - wB - wC - wD) * (overlap_hi - overlap_lo) / 3.5
            raw.append(0.50 * p + 0.50 * mix[b])
        total = sum(raw)
        units = [int(MICRO * x / total) for x in raw]
        left = MICRO - sum(units)
        order = sorted(range(n_bins), key=lambda b: (raw[b] / total * MICRO - units[b], -b), reverse=True)
        for b in order[:left]:
            units[b] += 1
        sizes.extend({"place": k, "bin": b, "p": units[b] / MICRO} for b in range(n_bins) if units[b])

    # --- top5: aggregate belief over places --------------------------------
    top5 = sorted(aggregate, key=lambda m: (-aggregate[m], rank_by_mint.get(m, 999)))[:places]
    while len(top5) < places:
        for row in chart:
            if row["mint"] not in top5:
                top5.append(row["mint"])
                break

    forecast = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / task["output"]).write_text(json.dumps(forecast, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
