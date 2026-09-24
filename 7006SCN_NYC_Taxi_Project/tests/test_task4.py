"""Task4 exporter runs with test-only aggregated fixture under pytest tmp_path.

NO synthetic test JSON is copied to the actual project results directory.
"""
from datetime import date

import pytest
from PIL import Image
from pyspark.sql import SparkSession

from coursework.settings import write_json
from coursework.task4 import create_contact_sheet, run_task4


@pytest.fixture(scope="module")
def spark():
    s = SparkSession.builder.master("local[1]").appName("SYNTHETIC_TASK4_ONLY").config(
        "spark.driver.memory", "512m").config("spark.sql.shuffle.partitions", "2").config(
        "spark.ui.enabled", "false").getOrCreate()
    s.sparkContext.setLogLevel("ERROR")
    yield s
    s.stop()


def test_exports_and_publication_gate(spark, tmp_path):
    synthetic = [
        (1.0, "Manhattan", "17", "2019-01", 6.0, 10.0),
        (0.0, "Queens", "9", "2019-11", 1.0, 10.0),
        (1.0, "Queens", "14", "2019-11", 4.0, 20.0),
    ]
    data = spark.createDataFrame(synthetic, ["label", "pickup_borough", "pickup_hour", "year_month", "trip_distance", "fare_amount"])
    data.write.mode("overwrite").parquet(str(tmp_path / "processed"))
    results = tmp_path / "results"; results.mkdir()
    task1 = {"status": "observed", "raw_row_count": 3, "clean_row_count": 3, "file_size_bytes": 100,
        "partition_count_before": 1, "partition_count_after_repartition": 2, "quality_counts_overlapping": {"invalid_fare": 0},
        "counts_by_month_and_class": [{"year_month": "2019-01", "label": 1.0, "count": 1},
                                     {"year_month": "2019-11", "label": 1.0, "count": 1},
                                     {"year_month": "2019-11", "label": 0.0, "count": 1}]}
    models = []
    for n in range(4):
        models.append({"name": f"Synthetic-{n}", "family": f"family-{n}", "cv_folds": 2,
            "cv_seconds": 1.0, "final_fit_seconds": 2.0,
            "threshold_tuning": {"threshold": 0.5},
            "tree_global_split_importance": [{"feature": "pickup_month_cyclic", "global_split_importance": 0.3}] if n in (1, 3) else [],
            "metrics": {**{name: 0.5 for name in
                ("auc_roc", "auc_pr", "accuracy", "positive_precision", "positive_recall", "positive_f1", "specificity")},
                "confusion": {"tp": 1, "tn": 1, "fp": 0, "fn": 0}}})
    task2 = {"status": "observed", "models": models}
    task3 = {"status": "observed", "explainability": {"feature_weights": [
        {"feature": f"f{i}", "local_weight": 0.1} for i in range(6)]},
        "fairness": {"groups": [{"borough": "Queens", "n": 2, "observed_positive_rate": 0.5,
                                  "predicted_positive_rate": 0.5, "tpr": 1.0, "fpr": 0.0, "precision": 1.0}]},
        "optimisations": [{"change": "shuffle_partition_count", "partitions": 2, "median_seconds": 1.0},
            {"change": "persist_MEMORY_AND_DISK", "median_before_seconds": 2.0,
             "median_after_seconds": 1.0, "cache_fill_seconds": 0.5}],
        "scalability_query_benchmarks": [{"sample_fraction": 0.5, "rows": 2, "group_by_seconds": 1.0}]}
    curves = {"curves": [{"model": m["name"], "points": [
        {"fpr": 0.0, "tpr": 0.0, "recall": 0.0, "precision": 1.0, "threshold": 1.0}]} for m in models]}
    for name, payload in (("task1.json", task1), ("task2.json", task2), ("task3.json", task3), ("task2_curves.json", curves)):
        write_json(results / name, payload)
    cfg = {"_project_root": str(tmp_path),
        "student": {"name": "Synthetic tester", "email": "tester@coventry.ac.uk", "sid": "TEST"},
        "allocation": {"pool_reference": "TEST-ONLY", "confirmed_against_aula_register": True,
            "licence_checked": True, "licence_or_terms_url": "https://example.org/test-only"},
        "paths": {"results_dir": "results", "processed_dir": "processed"},
        "model": {"minimum_borough_size": 1},
        "tableau": {"public_workbook_url": "", "published_four_dashboards_verified": False,
            "hourly_cluster_cost_usd": None, "cluster_cost_source": ""}}
    export = run_task4(spark, cfg)
    assert export["status"] == "pending_publication_or_screenshots"
    assert export["dashboards"] == []
    assert (results / "tbl_d2_performance.csv").exists()
    assert "groupby_fraction" in (results / "tbl_d4_scalability.csv").read_text(encoding="utf-8-sig")
    assert (results / "task4.json").exists()
    assert export["contact_sheet"] is None  # no fake screenshots!


def test_contact_sheet_requires_four_real_image_files(tmp_path):
    assert create_contact_sheet(tmp_path) is None
    for i in range(1, 5):
        Image.new("RGB", (80, 80), color=(i * 25, 40, 50)).save(tmp_path / f"tableau_dashboard_{i}.png")
    path = create_contact_sheet(tmp_path)
    assert path and (tmp_path / "task4_dashboard_contact_sheet.png").exists()
