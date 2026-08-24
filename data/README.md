# Data

`raw/` is read-only and never overwritten. `processed/` holds cleaned output.

The default config uses the synthetic generator in `src/lob_alpha/data/synthetic.py`, so the pipeline runs with no data licence. For real data, point the config at a directory of CSV or Parquet files and supply a `column_map` if the vendor schema differs from the canonical one:

    timestamp, instrument, bid_price, bid_size, ask_price, ask_size

Deeper levels are optional and follow the pattern `bid_price_2`, `bid_size_2`, and so on.

Every cleaning rule that drops a row records the count and reason in `results/tables/<experiment>/cleaning_audit.csv`. The data section of the paper is assembled from that file and `data_quality.csv`, so "the data were cleaned" never has to appear in the write-up.
