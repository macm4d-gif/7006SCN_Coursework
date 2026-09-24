"""Settings and provenance helpers shared by the four notebooks.

AI-assisted draft: inspect, adapt and test this module yourself; declare use in the report.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_TEMPLATE = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2019-{month:02d}.parquet"
ZONE_SOURCE = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"


def load_config(file: str = "config/config.json") -> dict[str, Any]:
    path = Path(file)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Copy config/config.example.json to config/config.json, "
            "then fill in YOUR allocation and your Spark cluster settings."
        )
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["_project_root"] = str(PROJECT_ROOT)
    return cfg


def project_path(cfg: dict, key: str) -> Path:
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else Path(cfg["_project_root"]) / p


def require_verified_allocation(cfg: dict) -> None:
    """Manual declaration, NOT independent verification of the read-only Aula register."""
    allocation = cfg["allocation"]
    student = cfg["student"]
    if not allocation.get("confirmed_against_aula_register", False):
        raise ValueError("Check the Aula allocation row, then explicitly confirm it in config/config.json.")
    if not allocation.get("licence_checked", False):
        raise ValueError("Read the assigned dataset's real usage terms and confirm licence_checked.")
    if not allocation.get("licence_or_terms_url", "").startswith("https://"):
        raise ValueError("Enter a verified licence/terms HTTPS URL (not a guessed licence title).")
    if not allocation.get("pool_reference") or "REPLACE" in allocation["pool_reference"]:
        raise ValueError("Pool reference must match YOUR allocation row exactly.")
    if any("REPLACE" in str(student.get(key, "")) for key in ("name", "email", "sid")):
        raise ValueError("Fill in your actual student identity in your untracked config.")
    if not student.get("email", "").endswith("@coventry.ac.uk"):
        raise ValueError("Check the email attached to your dataset allocation.")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict) -> None:
    """Atomic write; fail rather than silently serialising NaNs into invalid JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def check(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): check(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [check(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Non-finite metric: {value}")
        if hasattr(value, "item"):
            return check(value.item())
        return value

    data = json.dumps(check(payload), indent=2, ensure_ascii=False, allow_nan=False)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Run the preceding notebook first; missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def make_spark(cfg: dict, app: str):
    from pyspark.sql import SparkSession

    scfg = cfg["spark"]
    builder = SparkSession.builder.appName(f"7006SCN_{app}_NYC_Taxi")
    if scfg.get("master"):
        builder = builder.master(scfg["master"])
    builder = (
        builder.config("spark.driver.memory", scfg["driver_memory"])
        .config("spark.executor.memory", scfg["executor_memory"])
        .config("spark.executor.cores", int(scfg["executor_cores"]))
        .config("spark.sql.shuffle.partitions", int(scfg["shuffle_partitions"]))
        .config("spark.sql.session.timeZone", scfg["timezone"])
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.ui.enabled", "true")
    )
    # Only fix the worker count if the university allocation/cluster manager allows it.
    # Do not copy the reference notebook's unverified 4 executors verbatim.
    if scfg.get("executor_instances") is not None:
        instances = int(scfg["executor_instances"])
        if instances < 1:
            raise ValueError("spark.executor_instances must be a positive integer or null")
        builder = builder.config("spark.executor.instances", instances)
    if scfg.get("local_dir"):
        # Spark shuffle spill/scratch is node-local disk. Every executor must have
        # this directory (or the cluster manager can supply SPARK_LOCAL_DIRS).
        builder = builder.config("spark.local.dir", scfg["local_dir"])
    if scfg.get("event_log_dir"):
        folder = Path(scfg["event_log_dir"])
        folder.mkdir(parents=True, exist_ok=True)
        builder = builder.config("spark.eventLog.enabled", "true").config("spark.eventLog.dir", folder.as_uri())
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def spark_configuration(spark) -> dict:
    keys = (
        "spark.master", "spark.app.name", "spark.driver.memory", "spark.executor.memory",
        "spark.executor.instances", "spark.executor.cores", "spark.local.dir",
        "spark.sql.shuffle.partitions", "spark.sql.adaptive.enabled",
        "spark.sql.session.timeZone", "spark.eventLog.enabled", "spark.eventLog.dir",
    )
    conf = spark.sparkContext.getConf()
    return {key: conf.get(key, "not set") for key in keys} | {
        "spark_version": spark.version,
        "default_parallelism": spark.sparkContext.defaultParallelism,
        "application_id": spark.sparkContext.applicationId,
        "ui_url": spark.sparkContext.uiWebUrl,
    }
