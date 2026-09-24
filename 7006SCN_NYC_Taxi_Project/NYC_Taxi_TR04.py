#!/usr/bin/env python3
"""Readable, single-file CLI entry point for the numbered TR-04 Spark workflow.

This plays the same *role* as the user's Colab-exported .py file, without
copying the UK collision dataset, ordinal regression/RMSE or its printed scores.
The actual reusable transformations/CV/models live in coursework/*.py. The
FOUR numbered Task notebooks expose each stage, real outputs and plots.

The chronology is:
    1️⃣  Spark session (real university resources from a verified local config)
    2️⃣  All 12 official 2019 Yellow Taxi Parquet files and source validation
    3️⃣  Card-only >20% tip label, pickup-time features, missingness accounting
    4️⃣  Date+zone hash repartition, month-partitioned Parquet + Task 1 JSON
    5️⃣  Controlled cache/repartition/query benchmarks (Task 3)
    6️⃣  Fold-fitted feature pipeline and VectorAssembler (Task 2)
    7️⃣  Jan–Sep train/CV, Oct threshold, untouched Nov–Dec test
    8️⃣  Four Spark binary classifiers from >=3 families (NOT regressors)
    9️⃣  FOUR CrossValidators, test confusion/ROC/PR, measured training times
    🔟   SAVE and reload actual Spark PipelineModels, not an sklearn sample
    11. LIME, stability, Spark UI and subgroup analysis (Task 3)
    12. Aggregate CSVs and four real Tableau dashboards (Task 4)

The plan above maps to Task1.ipynb through Task4.ipynb. Execute their live cells
in the prescribed teaching weeks and keep real notebook output. This CLI also
runs the SAME tested coursework functions as a reproducible operational rerun;
it CANNOT manufacture student-authored results, weekly commits, dashboards,
screenshots or a final report. Do not run the full cohort on an undersized VM.

Examples (from the repository root; adjust approved spark-submit resources):
    python NYC_Taxi_TR04.py --plan
    spark-submit NYC_Taxi_TR04.py --config config/config.json --stage task1
    spark-submit NYC_Taxi_TR04.py --config config/config.json --stage task2
    spark-submit NYC_Taxi_TR04.py --config config/config.json --stage task3
    spark-submit NYC_Taxi_TR04.py --config config/config.json --stage task4
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGE_NAMES = ("task1", "task2", "task3", "task4")
PLAN = (
    "1️⃣  SPARK SESSION CONFIGURATION — real config and application ID",
    "2️⃣  DATA INGESTION + VALIDATION — twelve TLC 2019 monthly Parquets",
    "3️⃣  LABEL + DOMAIN FEATURES — card-only recorded tips; no outcome leakage",
    "4️⃣  PARTITIONING + PARQUET STORAGE — Spark QC and measured Task 1 JSON",
    "5️⃣  CACHING STRATEGY — benchmark benefit AND fill cost in Task 3",
    "6️⃣  VECTOR ASSEMBLY — fit preprocessing inside CV, not on full data",
    "7️⃣  TEMPORAL TRAIN / OCTOBER / FUTURE TEST SPLIT",
    "8️⃣  FOUR SPARK MLLIB CLASSIFIERS — linear, bagging, neural, boosting",
    "9️⃣  FOUR CROSSVALIDATORS + ROC/PR/CONFUSION/REAL TIMINGS",
    "🔟   MODEL SERIALIZATION — save and reload actual Spark pipelines",
    "11.  TASK 3 — Spark UI, scalability, four-model TEST perturbation, LIME and borough audit",
    "12.  TASK 4 — aggregate exports, genuine Tableau Public four-dashboard workbook",
)


def print_plan() -> None:
    """Show reference-style steps without pretending the university jobs ran."""
    print("TR-04 / NYC Yellow Taxi 2019 — executable workplan (NOT measured results)")
    for line in PLAN:
        print(line)
    print("Read notebooks/Task1.ipynb ... Task4.ipynb for the individual live code cells.")


def describe_result(stage: str, result: dict) -> None:
    """Print compact ACTUAL results from the just-completed assessed stage."""
    if stage == "task1":
        print("Source raw/eligible card rows:", result["raw_row_count"],
              result["clean_row_count"])
        print("Real source bytes and Parquet partitions:", result["file_size_bytes"],
              result["partition_count_after_repartition"])
        print("Staged Parquet:", result["processed_parquet_path"])
    elif stage == "task2":
        for row in result["models"]:
            print(row["name"], "CV s:", row["cv_seconds"],
                  "refit s:", row["final_fit_seconds"],
                  "Nov–Dec PR-AUC:", row["metrics"]["auc_pr"],
                  "ROC-AUC:", row["metrics"]["auc_roc"])
        print("Predeclared CV-based choice:", result["recommended_model"])
    elif stage == "task3":
        print("Measured optimisation arms:", len(result["optimisations"]))
        print("Four-model stability:", list(result["stability"]))
        print("Actual Spark-model local explanation:",
              result["explainability"]["method"])
    elif stage == "task4":
        print("Tableau aggregate extracts:", result["exports"])
        print("Public four-dashboard verification:", result["link_check"])
        print("Contact sheet from real screenshots:", result["contact_sheet"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run REAL TR-04 stages; or print the numbered workplan")
    parser.add_argument("--plan", action="store_true", help="Show numbered workflow without Spark or config")
    parser.add_argument("--config", default="config/config.json")
    parser.add_argument("--stage", choices=[*STAGE_NAMES, "all"],
                        help="Which real stage(s) to run (required except with --plan)")
    parser.add_argument("--download", action="store_true",
                        help="Fetch official 2019 Parquet + zone lookup first; target MUST be shared on cluster")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Check allocation/Spark/worker files without running any Task")
    args = parser.parse_args(argv)
    if args.plan:
        print_plan()
        return
    if not args.stage:
        parser.error("--stage task1|task2|task3|task4|all is required unless --plan is used")

    # 1️⃣ Spark configuration, identity and allocation are fail-closed.
    sys.path.insert(0, str(ROOT))
    from coursework.settings import (load_config, make_spark, project_path,
                                     require_verified_allocation, spark_configuration)
    from coursework.data import monthly_paths, run_task1
    from coursework.task2 import run_task2
    from coursework.task3 import run_task3
    from coursework.task4 import run_task4

    cfg = load_config(args.config)
    require_verified_allocation(cfg)  # check YOUR Aula row and actual use terms first
    if args.download:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "download_tlc.py"),
                        "--config", args.config], cwd=str(ROOT), check=True)

    print("\n1️⃣ SPARK SESSION CONFIGURATION (actual university allocation)", flush=True)
    paths: list[str] = []
    if args.stage in ("task1", "all"):
        paths = monthly_paths(cfg)
        print("2️⃣ 2019 OFFICIAL FILE PREFLIGHT:", len(paths),
              "files in", project_path(cfg, "raw_dir"), flush=True)
    spark = make_spark(cfg, f"Run_{args.stage}")
    print("Actual Spark version/resources/application:", spark_configuration(spark), flush=True)
    try:
        # 2️⃣ Both driver and executors need the POSIX shared files; do not read
        # a Colab /content path or a file visible only to the notebook driver.
        if paths:
            needed = [paths[0], paths[-1], str(project_path(cfg, "zone_lookup"))]
            checks = min(32, max(1, spark.sparkContext.defaultParallelism))
            visible = spark.sparkContext.parallelize(range(checks), checks).map(
                lambda _: all(os.path.isfile(path) for path in needed)).collect()
            if not all(visible):
                raise FileNotFoundError("Some Spark workers cannot see shared raw files/zone lookup")
            print("Worker shared-file preflight:", sum(visible), "/", checks, flush=True)
        if args.preflight_only:
            print("Preflight passed. No Task was run and NO result JSON was generated.")
            return

        stages = {"task1": run_task1, "task2": run_task2,
                  "task3": run_task3, "task4": run_task4}
        targets = list(STAGE_NAMES) if args.stage == "all" else [args.stage]
        for stage in targets:
            print("\n===", stage.upper(), "— RUN REAL SPARK WORK ===", flush=True)
            result = stages[stage](spark, cfg)
            print("Finished", stage, "status:", result["status"], flush=True)
            describe_result(stage, result)
            if result["status"] != "observed":
                raise SystemExit(
                    f"{stage} is NOT submission-ready: {result['status']}. "
                    "Task 4 needs a genuine Tableau Public workbook and four real screenshots."
                )
        print("Code stages ran. You still need real notebook outputs, a Spark UI screenshot, "
              "your published Tableau workbook, private-org Git history and a reviewed Word report.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
