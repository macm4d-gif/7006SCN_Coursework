"""TLC ingestion, data-quality accounting and leakage-safe pickup-time features.

All large-table operations use Spark DataFrames. Raw 2019 monthly schemas are
normalised individually BEFORE unionByName so parquet schema drift is explicit.
AI-assisted starting point: audit source terms, allocation and every filter.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Iterable

from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.types import DoubleType, IntegerType, TimestampType

from .settings import SOURCE_TEMPLATE, now_utc, project_path, require_verified_allocation, spark_configuration, write_json

RAW_TYPES = {
    "VendorID": IntegerType(),
    "tpep_pickup_datetime": TimestampType(),
    "tpep_dropoff_datetime": TimestampType(),
    "passenger_count": DoubleType(),
    "trip_distance": DoubleType(),
    "RatecodeID": IntegerType(),
    "PULocationID": IntegerType(),
    "DOLocationID": IntegerType(),
    "payment_type": IntegerType(),
    "fare_amount": DoubleType(),
    "extra": DoubleType(),
    "mta_tax": DoubleType(),
    "tip_amount": DoubleType(),
    "tolls_amount": DoubleType(),
    "improvement_surcharge": DoubleType(),
    "total_amount": DoubleType(),
    "congestion_surcharge": DoubleType(),
}
REQUIRED_RAW = {"tpep_pickup_datetime", "PULocationID", "payment_type", "fare_amount", "tip_amount"}
# The target/late measurements are NEVER used in the model features vector.
FORBIDDEN_AT_PICKUP = {
    "tip_amount", "fare_amount", "total_amount", "trip_distance",
    "tpep_dropoff_datetime", "DOLocationID", "payment_type", "RatecodeID",
}
# TLC dictionary describes RatecodeID as the *final* code at the END of the trip;
# it cannot safely be treated as available at pickup.
CATEGORICAL_FEATURES = (
    "pickup_borough", "VendorID", "pickup_hour", "pickup_dow"
)
# A future month is unseen by a month StringIndexer fit on Jan-Sep. Circular
# encoding lets Nov/Dec enter as valid time-of-year values, not an unknown bucket.
NUMERIC_FEATURES = ("passenger_count", "pickup_month_sin", "pickup_month_cos")


def monthly_paths(cfg: dict, months: Iterable[int] = range(1, 13)) -> list[str]:
    raw_dir = project_path(cfg, "raw_dir")
    paths = [raw_dir / f"yellow_tripdata_2019-{int(m):02d}.parquet" for m in months]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Download the official TLC Parquet first: missing {missing}")
    return [str(p) for p in paths]


def hadoop_file_sizes(spark: SparkSession, paths: list[str]) -> list[dict]:
    """Hadoop FS metadata on the SAME actual files Spark will read."""
    fs_conf = spark._jsc.hadoopConfiguration()
    jpath = spark._jvm.org.apache.hadoop.fs.Path
    output = []
    for file in paths:
        p = jpath(file)
        fs = p.getFileSystem(fs_conf)
        if not fs.exists(p):
            raise FileNotFoundError(f"Spark's Hadoop filesystem cannot see {file}")
        size = int(fs.getFileStatus(p).getLen())
        output.append({"path": file, "bytes": size, "source_url": SOURCE_TEMPLATE.format(month=int(Path(file).stem[-2:]))})
    return output


def load_raw_2019(spark: SparkSession, cfg: dict, months: Iterable[int] = range(1, 13)) -> tuple[DataFrame, list[dict]]:
    months = tuple(months)
    paths = monthly_paths(cfg, months)
    datasets: list[DataFrame] = []
    schema_log: list[dict] = []
    for month, path in zip(months, paths):
        source = spark.read.parquet(path)
        names = {name.lower(): name for name in source.columns}
        missing = [key for key in REQUIRED_RAW if key.lower() not in names]
        if missing:
            raise ValueError(f"{path} missing required fields {missing}; inspect the actual TLC dictionary")
        columns = [
            (F.col(names[key.lower()]) if key.lower() in names else F.lit(None)).cast(dtype).alias(key)
            for key, dtype in RAW_TYPES.items()
        ]
        month_df = source.select(*columns, F.lit(int(month)).alias("source_file_month"))
        datasets.append(month_df)
        schema_log.append({"month": int(month), "original_column_count": len(source.columns), "original_schema": source.schema.simpleString()})
    union = datasets[0]
    for other in datasets[1:]:
        union = union.unionByName(other, allowMissingColumns=False)
    return union, schema_log


def load_zones(spark: SparkSession, cfg: dict) -> DataFrame:
    path = project_path(cfg, "zone_lookup")
    if not path.is_file():
        raise FileNotFoundError(f"Missing official TLC zone lookup: {path}")
    return (
        spark.read.option("header", True).csv(str(path))
        .select(F.col("LocationID").cast("int").alias("zone_id"), F.col("Borough").alias("pickup_borough"))
        .dropna(subset=["zone_id"])
        .dropDuplicates(["zone_id"])
    )


def quality_counts(raw: DataFrame) -> dict:
    pickup_2019 = F.year("tpep_pickup_datetime") == 2019
    card = F.col("payment_type") == 1
    good_fare = (F.col("fare_amount") > 0) & (~F.isnan("fare_amount"))
    good_tip = (F.col("tip_amount") >= 0) & (~F.isnan("tip_amount"))
    good_zone = F.col("PULocationID").isNotNull()
    flags = {
        "not_2019_pickup": (~pickup_2019) | F.col("tpep_pickup_datetime").isNull(),
        "not_credit_card": (~card) | F.col("payment_type").isNull(),
        "invalid_fare": (~good_fare) | F.col("fare_amount").isNull(),
        "invalid_tip": (~good_tip) | F.col("tip_amount").isNull(),
        "missing_pickup_zone": ~good_zone,
        "missing_passenger_count": F.col("passenger_count").isNull(),
        "invalid_passenger_count": (F.col("passenger_count") < 1) | (F.col("passenger_count") > 8),
        "missing_vendor_id": F.col("VendorID").isNull(),
        "missing_rate_code": F.col("RatecodeID").isNull(),
    }
    agg = [F.count(F.lit(1)).alias("raw_rows")]
    agg += [F.sum(F.when(expr, 1).otherwise(0)).cast("long").alias(name) for name, expr in flags.items()]
    row = raw.agg(*agg).first().asDict()
    # Conditions overlap. Never add these columns to infer removed-row count.
    return {key: int(value or 0) for key, value in row.items()}


def prepare_tlc(raw: DataFrame, zones: DataFrame) -> DataFrame:
    valid = raw.where(
        (F.year("tpep_pickup_datetime") == 2019)
        & (F.col("payment_type") == 1)
        & (F.col("fare_amount") > 0) & (~F.isnan("fare_amount"))
        & (F.col("tip_amount") >= 0) & (~F.isnan("tip_amount"))
        & F.col("PULocationID").isNotNull()
    )
    enriched = valid.join(F.broadcast(zones), valid.PULocationID == zones.zone_id, "left")
    enriched = enriched.withColumn("passenger_count", F.when(
        F.col("passenger_count").between(1, 8), F.col("passenger_count")
    ).otherwise(F.lit(None).cast("double")))
    result = enriched.select(
        (F.col("tip_amount") / F.col("fare_amount") > 0.20).cast("double").alias("label"),
        F.coalesce(F.col("pickup_borough"), F.lit("Unknown")).alias("pickup_borough"),
        F.coalesce(F.col("VendorID").cast("string"), F.lit("Unknown")).alias("VendorID"),
        F.coalesce(F.col("RatecodeID").cast("string"), F.lit("Unknown")).alias("RatecodeID"),
        F.col("passenger_count"),
        F.hour("tpep_pickup_datetime").cast("string").alias("pickup_hour"),
        F.dayofweek("tpep_pickup_datetime").cast("string").alias("pickup_dow"),
        F.month("tpep_pickup_datetime").cast("string").alias("pickup_month"),
        F.sin(F.month("tpep_pickup_datetime") * F.lit(2 * math.pi / 12)).alias("pickup_month_sin"),
        F.cos(F.month("tpep_pickup_datetime") * F.lit(2 * math.pi / 12)).alias("pickup_month_cos"),
        F.to_date("tpep_pickup_datetime").alias("pickup_date"),
        F.date_format("tpep_pickup_datetime", "yyyy-MM").alias("year_month"),
        F.col("PULocationID"),
        # Retained ONLY for Task4 retrospective DESCRIPTIVE aggregates; never in features.
        F.col("trip_distance"), F.col("fare_amount"), F.col("tip_amount"),
    )
    assert FORBIDDEN_AT_PICKUP.isdisjoint(set(CATEGORICAL_FEATURES) | set(NUMERIC_FEATURES))
    return result


def split_processed(data: DataFrame) -> tuple[DataFrame, DataFrame, DataFrame]:
    train = data.where(F.col("year_month") < "2019-10")           # Jan--Sep, CV and final fit
    validation = data.where(F.col("year_month") == "2019-10")     # October threshold selection
    test = data.where(F.col("year_month") >= "2019-11")           # Nov--Dec, untouched final test
    return train, validation, test


def load_processed(spark: SparkSession, cfg: dict) -> DataFrame:
    path = project_path(cfg, "processed_dir")
    if not path.exists():
        raise FileNotFoundError(f"Run Task1 first; missing processed dataset {path}")
    return spark.read.parquet(str(path))


def run_task1(spark: SparkSession, cfg: dict) -> dict:
    require_verified_allocation(cfg)
    paths = monthly_paths(cfg)
    file_sizes = hadoop_file_sizes(spark, paths)
    size = sum(f["bytes"] for f in file_sizes)
    first_source = spark.read.parquet(paths[0])
    print("Actual original January schema (the 12 monthly schemas are also logged):")
    first_source.printSchema()
    raw, schema_log = load_raw_2019(spark, cfg)
    print("Canonical union schema:")
    raw.printSchema()
    started = time.perf_counter()
    qc = quality_counts(raw)
    raw_month_counts = {int(row['source_file_month']): int(row['count'])
                        for row in raw.groupBy('source_file_month').count().collect()}
    clean = prepare_tlc(raw, load_zones(spark, cfg))
    counts = [x.asDict() for x in clean.groupBy("year_month", "label").count().collect()]
    cleaned_rows = sum(int(item["count"]) for item in counts)
    split_counts = {
        "cv_train_jan_sep": sum(int(x["count"]) for x in counts if x["year_month"] < "2019-10"),
        "threshold_oct": sum(int(x["count"]) for x in counts if x["year_month"] == "2019-10"),
        "final_nov_dec": sum(int(x["count"]) for x in counts if x["year_month"] >= "2019-11"),
    }
    if min(split_counts.values()) == 0:
        raise ValueError(f"One temporal split is empty: {split_counts}")
    cfg_n = int(cfg["spark"]["shuffle_partitions"])
    # Hash both date and zone; partitioning on just one popular borough would skew.
    shuffled = clean.repartition(cfg_n, "pickup_date", "PULocationID")
    outdir = project_path(cfg, "processed_dir")
    shuffled.write.mode("overwrite").partitionBy("year_month").parquet(str(outdir))
    from .models import make_preprocessing_pipeline
    prep = make_preprocessing_pipeline()
    # The official EP1 sheet asks for FITTED pipeline stages in the evidence
    # pack. Fit a bounded TRAIN-ONLY demonstrator AFTER staging Parquet; actual
    # Task2 CV still fits fresh preprocessing per fold, never on held-out rows.
    demo = (spark.read.parquet(str(outdir)).where(F.col("year_month") < "2019-10")
            .limit(1500).cache())
    try:
        demo_rows = demo.count()
        if demo_rows < 100:
            raise ValueError("Insufficient Jan-Sep rows to fit an honest EP1 preprocessing demonstrator")
        fitted_stage_names = [type(stage).__name__ for stage in prep.fit(demo).stages]
    finally:
        demo.unpersist()
    blueprint_stages = [type(stage).__name__ for stage in prep.getStages()]
    result = {
        "status": "observed", "generated_utc": now_utc(),
        "student": cfg["student"], "pool_reference": cfg["allocation"]["pool_reference"],
        "dataset_name": cfg["allocation"]["dataset_name"],
        "source_page": cfg["allocation"]["source_page"],
        "source_files": file_sizes,
        "licence_or_terms_url": cfg["allocation"]["licence_or_terms_url"],
        "licence_reviewed_by_student": True,
        "ethical_note_to_review": cfg["allocation"]["ethical_note"],
        "raw_schema_by_month": schema_log,
        "raw_january_print_schema": first_source._jdf.schema().treeString(),
        "raw_column_count": len(first_source.columns),
        "raw_rows_by_source_month": raw_month_counts,
        "raw_row_count": qc["raw_rows"],
        "clean_row_count": cleaned_rows,
        "file_size_bytes": size,
        "file_size_gib": round(size / (1024**3), 4),
        "source_file_count": len(file_sizes),
        "big_data_verification": {
            "rows_gte_10m": qc["raw_rows"] >= 10_000_000,
            "columns_gte_10": min(x["original_column_count"] for x in schema_log) >= 10,
            "bytes_gte_1_gib": size >= 1024**3,
        },
        "quality_counts_overlapping": qc,
        "counts_by_month_and_class": counts,
        "temporal_split_rows": split_counts,
        "target_definition": "payment_type == 1; tip_amount/fare_amount > 0.20, fare_amount > 0",
        "decision_point": "prediction at pickup for credit-card-only eligible cohort; exclude final RatecodeID, realised fare, tips, distance, drop-off and payment method from model features",
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "excluded_from_features": sorted(FORBIDDEN_AT_PICKUP),
        "preprocessing_stages": fitted_stage_names,
        "preprocessing_blueprint_stages": blueprint_stages,
        "preprocessing_fit_sample_rows": demo_rows,
        "preprocessing_fit_limit": 1500,
        "preprocessing_fit_scope": "preprocessing-only demonstrator fit on first <=1500 Jan-Sep Parquet rows; Task2 refits per CV fold/full train",
        "preprocessing_stage_count": len(fitted_stage_names),
        "partition_count_before": raw.rdd.getNumPartitions(),
        "partition_count_after_repartition": shuffled.rdd.getNumPartitions(),
        "partitioning_rationale": "hash pickup_date + pickup zone rather than just skewed zone",
        "processed_parquet_path": str(outdir),
        "processing_seconds": round(time.perf_counter() - started, 3),
        "spark_configuration": spark_configuration(spark),
    }
    result_path = project_path(cfg, "results_dir") / "task1.json"
    write_json(result_path, result)
    from .evidence import task1_evidence_pack
    task1_evidence_pack(result, result_path.parent / "task1_evidence.png")
    if not all(result["big_data_verification"].values()):
        raise ValueError("Dataset fails a mandatory requirement. See actual counts in results/task1.json; contact Module Leader.")
    print(f"Observed raw rows={qc['raw_rows']:,}, clean rows={cleaned_rows:,}, bytes={size:,}, fitted EP1 stages={fitted_stage_names}")
    return result
