"""Tiny source-file schema-drift tests; no real-result JSON or submission evidence."""
from datetime import datetime
from pathlib import Path
import shutil

import pytest
from pyspark.sql import SparkSession

from coursework.data import hadoop_file_sizes, load_raw_2019, load_zones, prepare_tlc, quality_counts


@pytest.fixture(scope="module")
def spark():
    s = (SparkSession.builder.master("local[1]").appName("SYNTHETIC_SCHEMA_DRIFT")
         .config("spark.driver.memory", "512m").config("spark.sql.shuffle.partitions", "2")
         .config("spark.ui.enabled", "false").getOrCreate())
    s.sparkContext.setLogLevel("ERROR")
    yield s
    s.stop()


def test_monthly_parquet_file_union_size_and_credit_filter(spark, tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for m in (1, 2):
        timestamp = datetime(2019, m, 1, 12)
        data = [(timestamp, 1 if m == 1 else 2, 1, 9.0, 2.0, 1.0, 1.0)]
        names = ["tpep_pickup_datetime", "payment_type", "PULocationID",
                 "fare_amount", "tip_amount", "passenger_count", "VendorID"]
        df = spark.createDataFrame(data, schema=names)
        if m == 1:
            df = df.withColumn("total_amount", df.fare_amount + df.tip_amount)
        directory = tmp_path / f"part_{m}"
        df.coalesce(1).write.mode("overwrite").parquet(str(directory))
        source = next(directory.glob("part-*.parquet"))
        shutil.copyfile(source, raw_dir / f"yellow_tripdata_2019-{m:02d}.parquet")
    (raw_dir / "taxi_zone_lookup.csv").write_text("LocationID,Borough,Zone\n1,Queens,Demo\n")
    cfg = {"_project_root": str(tmp_path), "paths": {"raw_dir": "raw", "zone_lookup": "raw/taxi_zone_lookup.csv"}}
    raw, schema_by_month = load_raw_2019(spark, cfg, months=[1, 2])
    assert raw.count() == 2 and len(schema_by_month) == 2
    assert "total_amount" in raw.columns  # missing February becomes null via explicit cast
    assert quality_counts(raw)["not_credit_card"] == 1
    source_files = [str(raw_dir / f"yellow_tripdata_2019-{m:02d}.parquet") for m in (1, 2)]
    assert sum(x["bytes"] for x in hadoop_file_sizes(spark, source_files)) > 0
    clean = prepare_tlc(raw, load_zones(spark, cfg)).collect()
    assert len(clean) == 1 and clean[0]["label"] == 1.0
    assert clean[0]["pickup_borough"] == "Queens"
    assert "pickup_month_sin" in clean[0].asDict()
