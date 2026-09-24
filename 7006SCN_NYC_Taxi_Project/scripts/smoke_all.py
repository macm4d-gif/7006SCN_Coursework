"""Optional small SYNTHETIC end-to-end CV smoke test. NEVER submit these values.

Does not write results/task*.json, Tableau sheets or claims about the allocated data.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pyspark.sql import SparkSession
from coursework.task2 import run_task2

ROOT = Path(__file__).resolve().parents[1]
source = json.loads((ROOT / "config" / "config.example.json").read_text())
source["_project_root"] = str(ROOT)
source["paths"]["processed_dir"] = "data/processed/SYNTHETIC_SMOKE_ONLY"
source["model"]["curve_bins"] = 8
source["model"]["tuning_fraction"] = 1.0
source["model"]["cv_parallelism"] = 1
spark = (SparkSession.builder.master("local[1]").appName("SYNTHETIC_SMOKE_ONLY")
         .config("spark.driver.memory", "512m").config("spark.sql.shuffle.partitions", "2")
         .config("spark.sql.adaptive.enabled", "false").config("spark.ui.enabled", "false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
try:
    rows = []
    for idx in range(150):
        month = "2019-01" if idx < 90 else ("2019-10" if idx < 120 else "2019-11")
        label = float(idx % 4 in (0, 1))
        day = date(2019, int(month[-2:]), 1) + timedelta(days=(idx % 20))
        angle = 2 * math.pi * int(month[-2:]) / 12
        rows.append((label, "Manhattan" if idx % 2 else "Queens", str(1 + idx % 2),
            str(1 + idx % 3), float(1 + idx % 4), str(idx % 24), str(1 + idx % 7),
            str(int(month[-2:])), math.sin(angle), math.cos(angle), day, month,
            10 + idx % 4, float(1 + idx % 10), 8.0, 1.6 if label else 0.0))
    data = spark.createDataFrame(rows, schema=["label", "pickup_borough", "VendorID", "RatecodeID",
        "passenger_count", "pickup_hour", "pickup_dow", "pickup_month", "pickup_month_sin", "pickup_month_cos", "pickup_date", "year_month",
        "PULocationID", "trip_distance", "fare_amount", "tip_amount"])
    path = ROOT / source["paths"]["processed_dir"]
    data.write.mode("overwrite").partitionBy("year_month").parquet(str(path))
    output = run_task2(spark, source, smoke=True)
    assert output["status"] == "smoke_test_only" and len(output["models"]) == 4
    print("SYNTHETIC SMOKE PASSED; outputs must never be reported as NYC TLC results.")
finally:
    spark.stop()
