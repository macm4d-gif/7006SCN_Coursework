"""Synthetic Task3 test-PERTURBATION checks; never assessed taxi evidence."""
from __future__ import annotations

from pyspark.sql import SparkSession, functions as F
import pytest

from coursework.task3 import perturb_test_features


@pytest.fixture(scope="module")
def spark():
    s = (SparkSession.builder.master("local[1]").appName("SYNTHETIC_TEST_PERTURBATION")
         .config("spark.driver.memory", "512m").config("spark.ui.enabled", "false")
         .config("spark.sql.shuffle.partitions", "2").getOrCreate())
    s.sparkContext.setLogLevel("ERROR")
    yield s
    s.stop()


def test_perturbation_is_reproducible_and_changes_only_pickup_inputs(spark):
    original = spark.createDataFrame([
        (i, "23" if i % 2 else "8", "Queens" if i % 2 else "Unknown",
         float(i % 2), 100 + i, "2019-11") for i in range(400)
    ], ["row_id", "pickup_hour", "pickup_borough", "label", "fare_amount", "year_month"])
    with pytest.raises(ValueError, match="test_perturbation_fraction"):
        perturb_test_features(original, seed=31, fraction=1.0)
    a = perturb_test_features(original, seed=31, fraction=0.40).orderBy("row_id").collect()
    b = perturb_test_features(original, seed=31, fraction=0.40).orderBy("row_id").collect()
    assert a == b  # seed + same partitioning -> identical scenario
    changed = 0
    for row in a:
        before_hour = "23" if row.row_id % 2 else "8"
        before_borough = "Queens" if row.row_id % 2 else "Unknown"
        assert row.label == float(row.row_id % 2)
        assert row.fare_amount == 100 + row.row_id
        assert row.year_month == "2019-11"
        assert not (row._stability_hour_changed and row._stability_borough_changed)
        if row._stability_hour_changed:
            assert row.pickup_hour == ("0" if before_hour == "23" else "9")
            assert row.pickup_borough == before_borough
            changed += 1
        elif row._stability_borough_changed:
            assert row.pickup_hour == before_hour
            assert before_borough != "Unknown" and row.pickup_borough == "Unknown"
            changed += 1
        else:
            assert row.pickup_hour == before_hour
            assert row.pickup_borough == before_borough or (before_borough == "Unknown" and row.pickup_borough == "Unknown")
    assert 50 < changed < 160  # a real subset changed, not ALL rows or only one
