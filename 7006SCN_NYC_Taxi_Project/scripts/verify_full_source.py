"""Read-only 12-month source check before assessed Task1 (NO result JSON written).

Useful for diagnosing actual NYC 2019 schemas and whether a cluster sees all files.
This only checks source compatibility; results still require running Task1 yourself
with YOUR student identity, verified allocation, licence and live Spark evidence.

Example on a small local environment (not suitable for model CV):
  python scripts/verify_full_source.py --config config/config.example.json \
    --master 'local[1]' --driver-memory 768m --shuffle-partitions 8
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coursework.settings import load_config, make_spark
from coursework.data import (hadoop_file_sizes, load_raw_2019, load_zones,
                             monthly_paths, prepare_tlc, quality_counts)
from pyspark.sql import functions as F


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.example.json")
    parser.add_argument("--master", help="Optional temporary Spark master override")
    parser.add_argument("--driver-memory", help="Optional temporary driver memory override")
    parser.add_argument("--shuffle-partitions", type=int, help="Optional temporary shuffle count")
    parser.add_argument("--local-dir", help="Optional per-node scratch disk (not a shared data path)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    for flag, key in ((args.master, "master"), (args.driver_memory, "driver_memory"),
                      (args.shuffle_partitions, "shuffle_partitions"), (args.local_dir, "local_dir")):
        if flag is not None:
            cfg["spark"][key] = flag
    spark = make_spark(cfg, "ReadOnlySourceCheck")
    try:
        paths = monthly_paths(cfg)
        byte_count = sum(row["bytes"] for row in hadoop_file_sizes(spark, paths))
        print("Actual 12 downloaded file bytes (Hadoop FS):", byte_count, round(byte_count / 1024**3, 4), "GiB", flush=True)
        raw, schemas = load_raw_2019(spark, cfg)
        print("Raw columns by month:", [(row["month"], row["original_column_count"]) for row in schemas], flush=True)
        print("Canonical schema:")
        raw.printSchema()
        start = time.perf_counter()
        raw_quality = quality_counts(raw)
        print("Actual raw quality / count:", raw_quality, "seconds:", round(time.perf_counter()-start, 2), flush=True)
        start = time.perf_counter()
        cleaned = prepare_tlc(raw, load_zones(spark, cfg))
        by_month_class = [x.asDict() for x in cleaned.groupBy("year_month", "label").count()
                          .orderBy("year_month", "label").collect()]
        print("Actual card-only cleaned rows by month/class:", by_month_class,
              "seconds:", round(time.perf_counter()-start, 2), flush=True)
        print("MANDATORY DATA CHECK (source only):", {
            "raw_rows_gte_10m": raw_quality["raw_rows"] >= 10_000_000,
            "every_month_gte_10_columns": all(x["original_column_count"] >= 10 for x in schemas),
            "downloaded_bytes_gte_1_gib": byte_count >= 1024**3,
        }, flush=True)
        print("READ-ONLY SOURCE TEST. No assessed results files or Notebook evidence created.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
