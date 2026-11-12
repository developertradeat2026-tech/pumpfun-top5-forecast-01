#!/usr/bin/env python3
"""Deterministic incumbent/entrant forecaster for pumpfun-top5-forecast-01."""

import json
import math
import os
from datetime import datetime, timezone
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
        + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    return r if x >= 0 else 2.0 - r


def above(edge, mean, sd):
    if edge == -math.inf:
        return 1.0
    if edge == math.inf:
        return 0.0
    return 0.5 * erfc((edge - mean) / (sd * SQRT2))


def size_distribution(place, cap_sol, horizon, edges):
    """Lognormal cap uncertainty, widened for the multi-hour final horizon."""
    mean = math.log10(max(cap_sol, 1e-12))
    sd = 0.40 + 0.055 * math.sqrt(max(0.0, horizon - 1.0))
    raw = []
    for b in range(len(edges) + 1):
        lo = -math.inf if b == 0 else edges[b - 1]
        hi = math.inf if b == len(edges) else edges[b]
        raw.append(max(0.0, above(lo, mean, sd) - above(hi, mean, sd)))
    total = sum(raw)
    raw = [x / total for x in raw]
    # Largest remainder quantization preserves a total mass of exactly 1.
    units = [int(MICRO * x) for x in raw]
    left = MICRO - sum(units)
    order = sorted(range(len(raw)), key=lambda b: (MICRO * raw[b] - units[b], -b), reverse=True)
    for b in order[:left]:
        units[b] += 1
    return [
        {"place": place, "bin": b, "p": units[b] / MICRO}
        for b in range(len(units)) if units[b]
    ]


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
                row["cap"] = int(row["market_cap_lamports"])
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
    if reads:
        # One output covers every read. An extra horizon read at the midpoint
        # accounts for the mean scoring rule without over-weighting its tail.
        horizon_hours = max(0.0, (sum(reads) / len(reads) - cut) / 3600.0)
    else:
        horizon_hours = 1.0
    elapsed_hours = max(0.0, (cut - t0) / 3600.0)
    # After launch-flow saturation, incumbents deserve more mass; at 7+ hours
    # out, however, part of each place must remain for unseen launch winners.
    flow_ramp = min(1.0, elapsed_hours / 6.0)
    late_relax = min(0.20, 0.065 * max(0.0, horizon_hours - 2.0))
    place_mass = 0.58 + 0.12 * flow_ramp - late_relax
    place_mass = max(0.42, min(0.70, place_mass))
    rank_decay = 0.65 + 0.12 * flow_ramp - 0.035 * min(6.0, horizon_hours)
    rank_decay = max(0.36, min(0.80, rank_decay))

    previous_ranks = {}
    for mark in marks[:-1]:
        for row in mark["rows"]:
            previous_ranks[row["mint"]] = row["rank"]
    appearances = {}
    for mark in marks:
        for row in mark["rows"]:
            appearances[row["mint"]] = appearances.get(row["mint"], 0) + 1

    probabilities = []
    sizes = []
    aggregate = {}
    for place in range(1, places + 1):
        weights = {}
        for row in chart[:35]:
            rank = row["rank"]
            age_hours = max(0.0, (cut - timestamp(row["created_at"])) / 3600.0)
            survival = 0.40 + 0.60 * (1.0 - 2.0 ** (-age_hours / (20.0 / 60.0)))
            weight = math.exp(-rank_decay * abs(rank - place)) * survival
            if row.get("venue") == "pool":
                weight *= 1.20
            if appearances.get(row["mint"], 0) > 1:
                weight *= 1.15
            old_rank = previous_ranks.get(row["mint"])
            if old_rank is not None:
                velocity = max(-0.25, min(0.25, 0.04 * (old_rank - rank)))
                weight *= math.exp(velocity)
            weights[row["mint"]] = weight
        total_weight = sum(weights.values())
        candidates = list(weights.items())
        candidates.sort(key=lambda item: item[1], reverse=True)
        candidates = candidates[:35]
        floor_weight = total_weight * 0.00002 / place_mass
        candidates = [(m, w) for m, w in candidates if w >= floor_weight]
        candidate_total = sum(w for _, w in candidates)
        for mint, weight in candidates:
            p = place_mass * weight / candidate_total
            probabilities.append({"mint": mint, "place": place, "p": p})
            aggregate[mint] = aggregate.get(mint, 0.0) + p
        cap = chart[place - 1]["cap"] / 1e9
        sizes.extend(size_distribution(place, cap, horizon_hours, task["size"]["edges"]))

    rank_by_mint = {row["mint"]: row["rank"] for row in chart}
    top5 = sorted(aggregate, key=lambda m: (-aggregate[m], rank_by_mint[m]))[:places]
    forecast = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / task["output"]).write_text(json.dumps(forecast, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
