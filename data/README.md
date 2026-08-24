# Data

Two directories. `raw/` holds source files and is read-only. `processed/` holds cleaned output written by the pipeline.

The default config generates its data from `src/lob_alpha/data/synthetic.py`, so the pipeline runs end to end without a data licence.

## Using real data

Point the config at a directory of CSV or Parquet files. The canonical schema is:

```
timestamp, instrument, bid_price, bid_size, ask_price, ask_size
```

Deeper levels follow the pattern `bid_price_2`, `bid_size_2`, and so on, and are optional. If your vendor uses different column names, supply a `column_map` in the config and the loader will rename them.

LOBSTER exports load directly through `source: lobster`, which pairs the message and orderbook files by filename and handles the tick divisor and sentinel prices.

## Cleaning audit

Each cleaning rule that drops a row writes its count and reason to `results/tables/<experiment>/cleaning_audit.csv`. Together with `data_quality.csv`, that file gives the row budget for the data section of the paper, down to the individual rule.
