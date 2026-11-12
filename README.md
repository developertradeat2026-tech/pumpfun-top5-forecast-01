# pumpfun-top5-forecast-01

Deterministic, standard-library-only forecaster for the make.fun pump.fun top-5 bounty.

## Run

`python3 main.py` reads `/input/history.ndjson` and `/input/task.json`, and writes `/output/forecast.json`.

For local development:

`MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=output python3 main.py`

## Model

The model estimates each incumbents probability of occupying each place from the latest chart rank, observed longevity, repeated appearances, recent rank movement and graduation venue. It explicitly leaves probability mass for unnamed coins, including launches after the cut.

Market-cap distributions use a log-normal distribution centered on each place's recent market cap with conservative diffusion. Each bin is normalized and quantized with a largest-remainder method. The output is a single forecast for the seven answer reads.

No network access, external packages, random sampling or wall-clock access is used.

License: MIT.
