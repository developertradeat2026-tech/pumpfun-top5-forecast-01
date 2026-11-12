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
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
MICRO = 1_000_000
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
        p = 0.0
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
    prev = marks[-2] if len(marks) >= 2 else None
    prev_caps = {}
    prev_ranks = {}
    dt = 1.0
    if prev is not None:
        prev_caps = {row["mint"]: row["cap"] for row in prev["rows"]}
        prev_ranks = {row["mint"]: row["rank"] for row in prev["rows"]}
        dt = max(1e-9, (cut - timestamp(prev["at"])) / 3600.0)
    appearances = {}
    for mark in marks:
        for row in mark["rows"]:
            appearances[row["mint"]] = appearances.get(row["mint"], 0) + 1

    candidates = {}
    union = {row["mint"] for row in chart}
    if prev:
        union |= {row["mint"] for row in sorted(prev["rows"], key=lambda r: r["rank"])[:120]}
    for mint in union:
        row = next((r for r in chart if r["mint"] == mint), None)
        if row is None:
            prow = next((r for r in prev["rows"] if r["mint"] == mint), None)
            if prow is None:
                continue
            row = dict(prow)
            row["cap"] = prev_caps.get(mint, prow["cap"])
            row["rank"] = 200
        cnow = row["cap"]
        cprev = prev_caps.get(mint, cnow)
        vlog = math.log10(max(cnow, 1.0)) - math.log10(max(cprev, 1.0))
        vhat = max(-2.0, min(2.5, vlog / dt))
        age_h = max(0.0, (cut - timestamp(row["created_at"])) / 3600.0)
        candidates[mint] = {
            "rank": row["rank"],
            "cap": cnow,
            "vhat": vhat,
            "age": age_h,
            "venue": row.get("venue", "curve"),
            "seen": appearances.get(mint, 1),
        }

    # known-coin mass per place: short horizons keep more incumbents, long
    # horizons hand the places to coins born after the cut.
    s_known = 0.52 - 0.013 * max(0.0, min(h - 1.5, 9.0))
    s_known = max(0.40, min(0.55, s_known))
    lam = max(0.14, 0.22 - 0.009 * max(0.0, h - 1.5))
    # Naming a coin below the unnamed share it would otherwise receive is
    # self-defeating: name it well or leave it unnamed. Expected read chart
    # holds ~70-80 ids we cannot name (fresh launches), so the unnamed
    # share lands near (1-s_known)/78.
    q_min = max(0.004, (1.0 - s_known) / 78.0)

    probabilities = []
    aggregate = {}
    rank_by_mint = {row["mint"]: row["rank"] for row in chart}
    for place in range(1, places + 1):
        weights = {}
        for mint, feat in candidates.items():
            w = math.exp(-lam * abs(feat["rank"] - place))
            w *= math.exp(0.45 * feat["vhat"])
            if feat["venue"] == "pool":
                w *= 1.18
            elif feat["age"] < 4.0:
                w *= 1.30
            w *= 1.0 + 0.05 * min(feat["seen"], 12)
            weights[mint] = w
        total = sum(weights.values())
        named = []
        junk = []
        for mint, w in weights.items():
            q = s_known * w / total
            cap_sol = candidates[mint]["cap"] / 1e9
            if q >= q_min:
                named.append({"mint": mint, "place": place, "p": q})
            elif cap_sol < 60.0:
                # deep curve junk: named at p=0 — these sit near the bottom
                # of the read chart when they survive at all, and striking
                # them from the unnamed pool fattens the share the fresh
                # launches (unnameable) receive.
                junk.append({"mint": mint, "place": place, "p": 0.0, "cap_sol": cap_sol})
        named.sort(key=lambda r: (-r["p"], r["mint"]))
        junk.sort(key=lambda r: (r["cap_sol"], r["mint"]))
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
    floor_level = logcaps[places - 1]
    mean_level = sum(logcaps) / len(logcaps)
    hw = min(h, 7.0)
    wA = 0.46 - 0.012 * hw
    wB = 0.16 + 0.007 * hw
    wC = 0.09 + 0.004 * hw
    wD = 0.16
    wE = max(0.08, 1.0 - wA - wB - wC - wD)
    sizes = []
    for k in range(1, places + 1):
        lk = 0.72 * logcaps[k - 1] + 0.28 * mean_level
        sdA = 0.22 + 0.020 * hw
        comp = [
            (wA, lk, sdA),
            (wB, floor_level + 0.10, 0.33 + 0.010 * hw),
            (wC, floor_level + (0.75 if k == 1 else 0.55), 0.42 + 0.010 * hw),
            (wD, floor_level - 0.55, 0.55),
        ]
        sizes.extend(mixture_bins(k, edges, comp, wE, 2.0, 4.9))

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
