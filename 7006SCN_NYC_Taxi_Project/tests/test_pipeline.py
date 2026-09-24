"""Small SYNTHETIC correctness tests; these numbers are not coursework evidence."""
import math
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from pyspark.sql import SparkSession, functions as F

from coursework.data import CATEGORICAL_FEATURES, FORBIDDEN_AT_PICKUP, NUMERIC_FEATURES, prepare_tlc
from coursework.evaluation import binned_curve, full_metrics, scored_frame, select_threshold
from coursework.models import (FEATURE_VECTOR_SIZE, build_model_specs,
                               make_preprocessing_pipeline, pipeline_for,
                               tree_feature_importance)
from coursework.task3 import fairness_by_borough, test_perturbation_analysis as analyse_test_perturbation


@pytest.fixture(scope="module")
def spark():
    session = (SparkSession.builder.master("local[1]").appName("SYNTHETIC_TESTS_NOT_ASSESSMENT")
        .config("spark.driver.memory", "512m").config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.adaptive.enabled", "false").config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "America/New_York").getOrCreate())
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def synthetic_records(spark, n=100):
    base = datetime(2019, 1, 1, 12)
    records = []
    for i in range(n):
        when = base + timedelta(hours=i)
        # Actual pickup-time predictors available. A weak deterministic pattern
        # plus noise means both target classes in both day-group CV folds.
        label = float((i % 4) in (0, 1))
        angle = 2 * math.pi * (1 + i % 2) / 12
        records.append((label, "Manhattan" if i % 2 else "Queens", str(1 + i % 2),
                        str(1 + i % 3), float(1 + i % 4), str(i % 24), str(1 + i % 7),
                        str(1 + i % 2), math.sin(angle), math.cos(angle), when.date(), "2019-01", 100 + i % 3,
                        1.0 + i % 10, 8.0, 1.6 if label else 0.0, 1.0))
    return spark.createDataFrame(records, schema=["label", "pickup_borough", "VendorID", "RatecodeID",
        "passenger_count", "pickup_hour", "pickup_dow", "pickup_month", "pickup_month_sin", "pickup_month_cos", "pickup_date", "year_month",
        "PULocationID", "trip_distance", "fare_amount", "tip_amount", "class_weight"])


def test_target_is_card_only_and_excludes_leakage(spark):
    raws = [
        (datetime(2019, 1, 1, 12), 1, 2.0, 12.0, 3.0, 3.0, 1, 10, 1.0),
        (datetime(2019, 1, 2, 14), 2, 2.0, 12.0, 3.0, 3.0, 1, 10, 1.0),
        (datetime(2019, 1, 3, 14), 1, -2.0, 12.0, 3.0, 3.0, 1, 10, 1.0),
    ]
    df = spark.createDataFrame(raws, ["tpep_pickup_datetime", "payment_type", "fare_amount", "trip_distance",
                                     "tip_amount", "total_amount", "PULocationID", "VendorID", "passenger_count"])
    df = df.withColumn("RatecodeID", F.lit(1))
    zone = spark.createDataFrame([(1, "Queens")], ["zone_id", "pickup_borough"])
    output = prepare_tlc(df, zone).collect()
    assert len(output) == 1
    assert output[0]["label"] == 1.0  # 3 / 2 > 20%
    assert set(CATEGORICAL_FEATURES + NUMERIC_FEATURES).isdisjoint(FORBIDDEN_AT_PICKUP)


def test_four_families_real_spark_fit_and_metrics(spark):
    data = synthetic_records(spark)
    stages = make_preprocessing_pipeline().getStages()
    assert len(stages) >= 3
    assembler = next(stage for stage in stages if type(stage).__name__ == "VectorAssembler")
    assert not (set(assembler.getInputCols()) & FORBIDDEN_AT_PICKUP)
    specs = build_model_specs(smoke=True)
    assert {'linear', 'tree_ensemble', 'neural'} <= {spec.family for spec in specs}
    assert FEATURE_VECTOR_SIZE == 53
    for spec in specs:
        grid = spec.param_grid[0]
        estimator = spec.estimator.copy(grid)
        model = pipeline_for(estimator).fit(data)
        assert model.transform(data).select('features').first()['features'].size == FEATURE_VECTOR_SIZE
        if spec.name in ("RandomForestClassifier", "GBTClassifier"):
            grouped = tree_feature_importance(model, data)
            assert grouped and sum(row["global_split_importance"] for row in grouped) == pytest.approx(1.0)
        scored = scored_frame(model, data, spec.name).cache()
        try:
            points, info = binned_curve(scored, bins=6)
            threshold = select_threshold(scored, bins=6)
            metrics = full_metrics(scored, threshold["threshold"])
            assert metrics["rows"] == 100
            assert 0 <= metrics["auc_pr"] <= 1
            assert len(points) > 1
            assert info["positive_count"] == 50
            assert sum(metrics["confusion"].values()) == 100
        finally:
            scored.unpersist()


def test_borough_rates_direction_and_group_suppression(spark):
    rows = [(1.0, 0.9, "Manhattan"), (0.0, 0.8, "Manhattan"),
            (1.0, 0.1, "Queens"), (0.0, 0.3, "Queens"),
            (1.0, 0.9, "Unknown")]
    df = spark.createDataFrame(rows, ["label", "score", "pickup_borough"])
    fairness = fairness_by_borough(df, threshold=0.5, minimum_size=2)
    assert fairness["min_over_max_predicted_positive_rate"] == 0
    assert {x["borough"] for x in fairness["groups"]} == {"Manhattan", "Queens"}


def test_all_four_saved_models_score_same_perturbed_holdout(spark, tmp_path):
    """Synthetic end-to-end test only: real saved Spark models, no assessed JSON."""
    train = synthetic_records(spark, n=120)
    holdout = synthetic_records(spark, n=120).withColumn("year_month", F.lit("2019-11"))
    data = train.unionByName(holdout)
    cfg = {"_project_root": str(tmp_path), "paths": {"artifact_dir": "artifacts"},
           "model": {"seed": 31, "test_perturbation_fraction": 0.40}}
    rows = []
    for spec in build_model_specs(smoke=True):
        classifier = spec.estimator.copy(spec.param_grid[0])
        fitted = pipeline_for(classifier).fit(train)
        save = tmp_path / "artifacts" / "models" / spec.name
        fitted.write().overwrite().save(str(save))
        baseline = full_metrics(scored_frame(fitted, holdout, spec.name), threshold=0.5)
        rows.append({"name": spec.name, "metrics": baseline,
                     "threshold_tuning": {"threshold": 0.5}})
    measured = analyse_test_perturbation(spark, cfg, rows, data)
    protocol = measured["protocol"]
    assert protocol["rows"] == 120 and protocol["labels_changed"] == 0
    assert protocol["positive_labels"] == 60 and protocol["same_test_rows_as_task2"]
    assert protocol["hour_shifted"] + protocol["borough_hidden"] > 0
    assert set(measured["models"]) == {r["name"] for r in rows}
    assert {r["rank"] for r in measured["models"].values()} == {1, 2, 3, 4}
    for name, info in measured["models"].items():
        assert info["fixed_october_threshold"] == 0.5
        assert abs(info["signed_deltas"]["auc_roc"]) <= 1.0
        assert abs(info["signed_deltas"]["positive_f1"]) <= 1.0
        assert (tmp_path / "artifacts" / "stability_predictions" / name).is_dir()
