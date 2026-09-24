"""Resource counts are cluster-specific: do not silently force the reference's four."""
import json
from pathlib import Path

import pytest

from coursework.settings import make_spark

ROOT = Path(__file__).resolve().parents[1]


def test_example_configs_leave_executor_count_to_cluster():
    for name in ("config.example.json", "config.university.example.json"):
        cfg = json.loads((ROOT / "config" / name).read_text(encoding="utf-8"))
        assert cfg["spark"]["executor_instances"] is None
        assert cfg["allocation"]["confirmed_against_aula_register"] is False


def test_invalid_executor_count_is_rejected_before_spark_starts():
    settings = {"master": "local[1]", "driver_memory": "1g", "executor_memory": "1g",
                "executor_cores": 1, "executor_instances": 0,
                "shuffle_partitions": 2, "timezone": "America/New_York"}
    with pytest.raises(ValueError, match="executor_instances"):
        make_spark({"spark": settings}, "DO_NOT_START")
