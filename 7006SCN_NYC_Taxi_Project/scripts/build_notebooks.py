"""Build the four mandatory, step-by-step TR-04 notebooks.

Presentation follows the user's numbered Spark notebook, but the real experiment
remains the *allocated* NYC Yellow Taxi classification task. Heavy Spark work is
implemented ONCE in coursework/ and called from these notebooks. All code cells
are unexecuted until the student runs them on their university cluster; no old
collision CSV, RMSE, fabricated outputs, or fake Tableau workbooks are included.

Run from the project root: python scripts/build_notebooks.py
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"
OUT.mkdir(exist_ok=True)
NUMBERS = {1: "1️⃣", 2: "2️⃣", 3: "3️⃣", 4: "4️⃣", 5: "5️⃣", 6: "6️⃣",
           7: "7️⃣", 8: "8️⃣", 9: "9️⃣", 10: "🔟"}


def build(name: str, title: str, introduction: str, steps: list[tuple[int, str, str, str]], conclusion: str) -> None:
    cells = [nbf.v4.new_markdown_cell(f"# {title}\n\n{introduction}")]
    assert [number for number, *_ in steps] == list(range(1, len(steps) + 1)), name
    assert len(steps) <= 10, name
    for number, heading, explanation, source in steps:
        cells.append(nbf.v4.new_markdown_cell(f"## {NUMBERS[number]} {heading}\n\n{explanation}"))
        cells.append(nbf.v4.new_code_cell(dedent(source).strip() + "\n"))
    cells.append(nbf.v4.new_markdown_cell(conclusion))
    book = nbf.v4.new_notebook(cells=cells)
    book.metadata["kernelspec"] = {"display_name": "Python 3 (TR-04 university Spark)",
                                    "language": "python", "name": "python3"}
    book.metadata["language_info"] = {"name": "python"}
    nbf.write(book, OUT / name)
    print(f"{name}: {len(steps)} numbered steps, {len(cells)} cells (UNEXECUTED)")


SETUP = '''\
from pathlib import Path
import sys
from pyspark.sql import SparkSession
ROOT = Path.cwd()
if not (ROOT / "coursework").is_dir():
    ROOT = ROOT.parent  # also works when Jupyter starts inside notebooks/
if not (ROOT / "coursework").is_dir():
    raise RuntimeError("Launch Jupyter from the TR-04 project root or notebooks/ directory")
sys.path.insert(0, str(ROOT))
from coursework.settings import (load_config, make_spark, project_path,
    read_json, require_verified_allocation, spark_configuration)
cfg = load_config()  # gitignored config/config.json: your real allocation and cluster
require_verified_allocation(cfg)  # requires YOUR independently checked Aula row and terms
spark = make_spark(cfg, "__TASK__")  # config-driven SparkSession.builder, not guessed Colab resources
assert isinstance(spark, SparkSession)
print("Student and allocated dataset:", cfg["student"], cfg["allocation"]["pool_reference"],
      cfg["allocation"]["dataset_name"])
print("ACTUAL Spark application / resources:", spark_configuration(spark))
'''


def setup(task: str) -> str:
    return SETUP.replace("__TASK__", task)


build(
    "Task1.ipynb", "Task 1 — 2019 NYC Yellow Taxi: ingest, validate and prepare",
    "**Student:** fill your own identity in the local config. **Pool:** TR-04 only after checking your Aula allocation. "
    "**Question:** Among eligible card trips, predict whether the *recorded* tip exceeds 20% of fare using "
    "only pickup-time inputs. Run the ten cells below **in order**, in a fresh kernel on your university Spark "
    "cluster. Results and figures remain empty in this template until you run the real 12-month source. "
    "Do not paste collision outputs into this notebook.",
    [
        (1, "SPARK SESSION CONFIGURATION (ACTUAL RESOURCES)",
         "`make_spark` uses `SparkSession.builder` with the university settings you provide in `config/config.json`; "
         "it prints what the live session actually received. A pre-existing SparkContext cannot be resized by "
         "editing a notebook cell: restart/re-submit with approved driver/executor resources. Do not assume the "
         "reference project's hard-coded 4×6 GB or 400 shuffle partitions are optimal here.",
         setup("Task1")),
        (2, "DATA INGESTION + SCHEMA/FILE VALIDATION",
         "Load **all 12 official 2019 TLC Parquet files**, not the UK collision CSV. Spark normalises monthly optional "
         "columns/types before `unionByName`. Source byte sizes come from the same Hadoop filesystem that Spark "
         "will read; verify driver **and workers** can see the shared POSIX files. The preview is five rows, not "
         "a fabricated total. Do not publish row-level data.",
         '''\
import os
from coursework.data import monthly_paths, hadoop_file_sizes, load_raw_2019
paths = monthly_paths(cfg)
source_files = hadoop_file_sizes(spark, paths)
assert len(paths) == len(source_files) == 12
needed = [paths[0], paths[-1], str(project_path(cfg, "zone_lookup"))]
workers_to_check = min(8, max(1, spark.sparkContext.defaultParallelism))
visible = spark.sparkContext.parallelize(range(workers_to_check), workers_to_check).map(
    lambda _: all(os.path.isfile(p) for p in needed)).collect()
if not all(visible):
    raise FileNotFoundError("Driver/worker shared-file preflight failed; fix cluster mounts")
print("Workers with required shared files:", sum(visible), "/", len(visible))
print("Actual source bytes:", sum(item["bytes"] for item in source_files))
print("Original January schema:")
spark.read.parquet(paths[0]).printSchema()
raw, source_schemas = load_raw_2019(spark, cfg)
print("Original field counts by file:",
      [(item["month"], item["original_column_count"]) for item in source_schemas])
print("Canonical 12-month Spark schema:")
raw.printSchema()
raw.select("source_file_month", "tpep_pickup_datetime", "PULocationID",
           "fare_amount", "tip_amount", "payment_type").limit(5).show(truncate=False)
'''),
        (3, "DOMAIN FEATURE ENGINEERING + VALID-LABEL FILTER",
         "Use a Spark join to the official taxi-zone lookup; target = `tip_amount / fare_amount > 0.20` "
         "**only for trips paid by card with valid observed fare/tip and 2019 pickup**. Cash tips are not "
         "recorded. Derive borough, hour, weekday and circular month from pickup. Final RatecodeID, payment type, "
         "fare, tip, realised distance and drop-off are NOT model inputs. The preview is a small live query; "
         "full counts are computed once by Step 4.",
         '''\
from coursework.data import (prepare_tlc, load_zones, CATEGORICAL_FEATURES,
    NUMERIC_FEATURES, FORBIDDEN_AT_PICKUP)
zones = load_zones(spark, cfg)
eligible = prepare_tlc(raw, zones)  # lazy Spark transformation
features_at_pickup = set(CATEGORICAL_FEATURES) | set(NUMERIC_FEATURES)
assert features_at_pickup.isdisjoint(FORBIDDEN_AT_PICKUP)
print("Pickup-time predictors:", sorted(features_at_pickup))
print("Excluded late/outcome fields:", sorted(FORBIDDEN_AT_PICKUP))
eligible.select("label", "pickup_borough", "pickup_hour", "pickup_dow",
                "pickup_month_sin", "pickup_month_cos").limit(5).show(truncate=False)
'''),
        (4, "PARTITIONING + PARQUET STORAGE (ONE AUTHORITATIVE RUN)",
         "This step runs all full-source quality counts, `repartition(shuffle_partitions, pickup_date, "
         "PULocationID)`, and one `write.partitionBy('year_month').parquet(...)` in `coursework/data.py`. "
         "The plan inspection is lazy; **do not duplicate the 60M-row Parquet write** in a second cell. "
         "The real `task1.json` records row counts, bytes, schema, partition settings and stage names.",
         '''\
from coursework.data import run_task1, load_processed
planned = eligible.repartition(int(cfg["spark"]["shuffle_partitions"]),
                               "pickup_date", "PULocationID")
print("Requested hash partitioning:", int(cfg["spark"]["shuffle_partitions"]),
      "over pickup_date + pickup zone (not a single skewed borough)")
planned.explain()  # no second full write or second full count
observed_1 = run_task1(spark, cfg)  # full 12-file Spark scan and Parquet write, once
processed = load_processed(spark, cfg)
print("Raw rows / retained eligible card trips:",
      observed_1["raw_row_count"], observed_1["clean_row_count"])
print("Recorded before/after partition counts:",
      observed_1["partition_count_before"], observed_1["partition_count_after_repartition"])
print("Partitioned Parquet location:", observed_1["processed_parquet_path"])
processed.printSchema()
'''),
        (5, "CACHING STRATEGY (BOUNDED AGGREGATES)",
         "Unlike the reference's unconditional full-data `persist()`, stage Parquet on shared storage and "
         "cache only a **small reused monthly aggregate** here. The first action fills it; the second reuses "
         "it; unpersist when done. Task 3 separately benchmarks whether caching the training data pays "
         "off **including cache-fill cost**.",
         '''\
from pyspark.storagelevel import StorageLevel
monthly_grouped = processed.groupBy("year_month", "label").count().persist(
    StorageLevel.MEMORY_AND_DISK)
try:
    print("Cached month × label aggregate groups:", monthly_grouped.count())
    monthly_pd = monthly_grouped.orderBy("year_month", "label").toPandas()  # <=24 rows
finally:
    monthly_grouped.unpersist(blocking=True)
from IPython.display import display
display(monthly_pd)
'''),
        (6, "VECTOR ASSEMBLY + FOLD-SAFE PREPROCESSING",
         "`coursework/models.py` defines fixed pickup-time `StringIndexerModel` vocabularies, "
         "plus learned `Imputer`, `OneHotEncoder`, `VectorAssembler` and scaler stages. Task1 "
         "fits an illustrative preprocessing-only PipelineModel on <=1,500 **Jan–Sep** rows "
         "for EP1; actual Task2 learns a NEW imputer/encoder/scaler **inside every CV fold** "
         "and in each full-training refit. Never fit transforms using test rows.",
         '''\
from pyspark.ml.feature import VectorAssembler
from coursework.models import make_preprocessing_pipeline
prep = make_preprocessing_pipeline()
for i, stage in enumerate(prep.getStages(), 1):
    print(f"Pipeline stage {i}: {type(stage).__name__}")
assembler = next(s for s in prep.getStages() if isinstance(s, VectorAssembler))
assert set(assembler.getInputCols()).isdisjoint(FORBIDDEN_AT_PICKUP)
print("Assembled, TRAIN-FITTED feature inputs:", assembler.getInputCols())
print("EP1 fitted TRAIN-ONLY demonstrator stages:", observed_1["preprocessing_stages"])
print("Task2 independently FITS per CV fold; no preprocessing is fitted on held-out data.")
'''),
        (7, "TRAIN / THRESHOLD / FUTURE TEST SPLIT",
         "January–September 2019 is for CV and full final fitting; October tunes each decision threshold; "
         "November–December remains untouched until one final evaluation. Do **not** use the reference "
         "project's random 80/20 split for time-sensitive taxi data. Counts below are measured in Step 4.",
         '''\
from coursework.data import split_processed
train, october, final_test = split_processed(processed)  # Spark DataFrames, lazy
splits = observed_1["temporal_split_rows"]
assert sum(splits.values()) == observed_1["clean_row_count"]
print("Measured Jan–Sep / October / Nov–Dec rows:", splits)
print("Fold candidates are calendar-day-grouped within Jan–Sep (not forward-chaining).")
'''),
        (8, "DATA QUALITY, SOURCE VOLUME + FIVE Vs",
         "Show actual original-file fields/bytes, monthly rows and **non-exclusive** data-quality conditions. "
         "Card-only filtering is an outcome-observation restriction, not just bad-data deletion. "
         "Monthly historic totals are NOT a measured streaming ingestion velocity.",
         '''\
import pandas as pd
from IPython.display import display
month_counts = pd.DataFrame([
    {"source_month": month, "raw_rows": count}
    for month, count in observed_1["raw_rows_by_source_month"].items()
]).sort_values("source_month")
display(month_counts)
quality_table = pd.DataFrame([
    {"condition (overlaps permitted)": name, "rows": count}
    for name, count in observed_1["quality_counts_overlapping"].items()
])
display(quality_table)
print("Original columns / compressed source GiB:", observed_1["raw_column_count"],
      observed_1["file_size_gib"])
print("Mandatory dataset checks:", observed_1["big_data_verification"])
'''),
        (9, "EDA PLOTS — MEASURED MONTHLY LABELS + BOROUGH RATES",
         "All heavy groupBy operations are Spark; only at most 24 month-label cells and a few borough "
         "aggregates are collected for matplotlib. Counts/rates describe the **eligible card-only cohort**, "
         "not a causal effect or a complete census of cash tips. Do not publish trip-level extracts.",
         '''\
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
monthly_pd["year_month"] = monthly_pd["year_month"].astype(str)
ax = monthly_pd.pivot(index="year_month", columns="label", values="count").fillna(0).plot(
    kind="bar", stacked=True, figsize=(11, 3.5), color=["#315c84", "#dc784d"])
ax.set(title="Eligible card trips: recorded label by pickup month",
       xlabel="2019 pickup month", ylabel="Trips")
plt.tight_layout()
plt.show()
borough_pd = (processed.groupBy("pickup_borough")
              .agg(F.count(F.lit(1)).alias("n"), F.avg("label").alias("high_tip_rate"))
              .where(F.col("n") >= int(cfg["model"]["minimum_borough_size"]))
              .orderBy("pickup_borough").toPandas())  # few aggregate boroughs
print("Boroughs meeting disclosure threshold:")
display(borough_pd)
ax = borough_pd.plot.bar(x="pickup_borough", y="high_tip_rate", legend=False,
                         figsize=(7, 3), color="#168a85")
ax.set(xlabel="Pickup borough", ylabel="Recorded >20% tip rate",
       title="Observed borough rates (card-only cohort; not causal)")
plt.tight_layout()
plt.show()
'''),
        (10, "ONE-CELL EP1 IMAGE + YOUR INTERPRETATION",
         "This SINGLE cell redraws one ordered composite from this run's measured SparkSession, "
         "original printSchema(), raw row/column counts, Hadoop file bytes, partition counts "
         "and ACTUAL FITTED Jan–Sep-only preprocessing stages. The demonstrator is NOT a "
         "full fitted Task2 classifier. Write your own five-V/ethics analysis and verify identity "
         "and allocated-source terms before committing.",
         '''\
from IPython.display import Image, display
from coursework.evidence import task1_evidence_pack
assert observed_1["status"] == "observed"
assert all(observed_1["big_data_verification"].values())
evidence = project_path(cfg, "results_dir") / "task1_evidence.png"
task1_evidence_pack(observed_1, evidence)  # one cell -> one EP1 composite image
display(Image(filename=str(evidence)))
print("Measured provenance/result:", project_path(cfg, "results_dir") / "task1.json")
print("To explain in YOUR report: 5 Vs, overlap in quality flags, card-only population,")
print("monthly seasonal patterns, why final outcome fields are not model features.")
'''),
    ],
    "**Task 1 completion:** review the actual output and report caveats, commit your genuinely observed progress "
    "to the required **private module-organisation** repository, then move to Task 2. No execution output "
    "or sample score is supplied in this template.",
)


build(
    "Task2.ipynb", "Task 2 — four tuned Spark ML classifiers and evaluation",
    "**Run after Task 1 on the actual staged Parquet.** The task is binary classification, not "
    "collision-severity regression: compare four classifiers from at least three families, tune all "
    "four, score the same untouched Nov–Dec test period and save real Spark PipelineModels. "
    "A fresh kernel/session helps ensure your actual cluster configuration is captured.",
    [
        (1, "SPARK SESSION CONFIGURATION + TASK 1 GATE",
         "Use the same verified student/allocation, approved resources and Spark version as your assessed "
         "Task 1 run. Settings requested inside an existing JVM may not take effect.",
         setup("Task2")),
        (2, "READ STAGED PARQUET + TIME SPLIT",
         "This is the **same** Task 1 output; no sampling of the final test. Use the real observed "
         "Task 1 counts rather than rerunning a huge `count()` simply to print an example number.",
         '''\
from coursework.data import load_processed, split_processed
meta = read_json(project_path(cfg, "results_dir") / "task1.json")
assert meta["status"] == "observed" and all(meta["big_data_verification"].values())
data = load_processed(spark, cfg)
train, october, final_test = split_processed(data)
print("Measured Jan–Sep / October / Nov–Dec:", meta["temporal_split_rows"])
print("Target:", meta["target_definition"])
print("Staged Parquet:", meta["processed_parquet_path"])
'''),
        (3, "CLASS BALANCE + DAY-GROUPED CV FOLDS",
         "Use training-only prevalence for inverse class weights on LR/RF/GBT. Spark 3.5 MLP "
         "does not support `weightCol`, so it is fitted unweighted on the same Jan–Sep rows "
         "and its different loss must be disclosed. Each pickup date belongs to one CV fold, "
         "but day-grouped CV is not rolling-origin validation. Thresholds use October only.",
         '''\
train_cells = [r for r in meta["counts_by_month_and_class"]
               if r["year_month"] < "2019-10"]
by_label = {label: sum(int(x["count"]) for x in train_cells if float(x["label"]) == label)
            for label in (0.0, 1.0)}
assert by_label[0.0] and by_label[1.0]
total_train = sum(by_label.values())
print("Jan–Sep class counts:", by_label)
print("Positive prevalence:", by_label[1.0] / total_train)
print("Training-only inverse class weights:",
      {label: total_train / (2 * n) for label, n in by_label.items()})
print("CV folds / tuning fraction:", cfg["model"]["cv_folds"],
      cfg["model"]["tuning_fraction"])
print("NOTE: Spark MLP has no weightCol; its fit is UNWEIGHTED on these same rows.")
'''),
        (4, "VECTOR ASSEMBLY + FOUR MLLIB CLASSIFIER FAMILIES",
         "Fixed pickup-time category vocabularies define the SAME vector width in every fold. "
         "Learned `Imputer`/`OneHotEncoder`/`StandardScaler` fit **inside** each CV Pipeline; "
         "VectorAssembler excludes late/label fields. Families: LR (linear), RF (tree ensemble), "
         "Spark MLP (neural) and GBT (free choice). Do not call LinearSVC a nonlinear kernel model.",
         '''\
from pyspark.ml.classification import (LogisticRegression, RandomForestClassifier,
    MultilayerPerceptronClassifier, GBTClassifier)
from pyspark.ml.feature import VectorAssembler
from coursework.models import FEATURE_VECTOR_SIZE, make_preprocessing_pipeline, build_model_specs
from coursework.data import FORBIDDEN_AT_PICKUP
prep = make_preprocessing_pipeline()
assembler = next(stage for stage in prep.getStages() if isinstance(stage, VectorAssembler))
assert set(assembler.getInputCols()).isdisjoint(FORBIDDEN_AT_PICKUP)
specs = build_model_specs()
assert len(specs) == 4 and {'linear', 'tree_ensemble', 'neural'} <= {s.family for s in specs}
for s in specs:
    print("MODEL:", s.name, "FAMILY:", s.family, "WHY:", s.rationale_to_check)
print("VectorAssembler inputs (late/outcome columns absent):", assembler.getInputCols())
print("Predeclared constant feature width for Spark neural network:", FEATURE_VECTOR_SIZE)
'''),
        (5, "DISTRIBUTED CROSS VALIDATION — EACH MODEL HAS A GRID",
         "`run_task2` constructs this `CrossValidator` blueprint **four times** with each actual "
         "param grid, `areaUnderPR` evaluator and day-grouped `foldCol`. The cell creates one "
         "*illustrative object only* (no fit); Step 6 does all four real CV fits. The reference "
         "project tuned only RF and used regression RMSE, which would not satisfy TR-04.",
         '''\
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from coursework.models import pipeline_for, param_grid_as_dicts
for s in specs:
    print(s.name, "candidate parameters:", param_grid_as_dicts(s))
first = specs[0]
cv_blueprint = CrossValidator(
    estimator=pipeline_for(first.estimator), estimatorParamMaps=first.param_grid,
    evaluator=BinaryClassificationEvaluator(labelCol="label",
        rawPredictionCol="rawPrediction", metricName="areaUnderPR"),
    numFolds=int(cfg["model"]["cv_folds"]), foldCol="cv_fold",
    parallelism=int(cfg["model"]["cv_parallelism"]),
)
print("Illustrative fold-aware CV blueprint (NOT FITTED HERE):",
      cv_blueprint.getNumFolds(), "folds")
'''),
        (6, "FIT FOUR CVs + FULL-TRAIN REFITS + REAL MODEL SAVES",
         "**This is the expensive assessed operation**: four CV grids on a documented Jan–Sep "
         "training-only sample, then one full-Jan–Sep best-param refit per model. Spark PipelineModels "
         "are actually saved to `artifacts/models/`; no arbitrary 50k-row sklearn comparison. "
         "Thresholds are chosen on October; Nov–Dec is scored once per fitted model.",
         '''\
from coursework.task2 import run_task2
observed_2 = run_task2(spark, cfg)  # ACTUAL four CrossValidator fits, refits and holdout scoring
assert observed_2["status"] == "observed" and len(observed_2["models"]) == 4
for row in observed_2["models"]:
    path = project_path(cfg, "artifact_dir") / "models" / row["name"]
    assert path.is_dir(), path
    print(row["name"], "CV folds:", row["cv_folds"],
          "CV seconds:", row["cv_seconds"], "full fit seconds:", row["final_fit_seconds"],
          "weighting:", row["training_weight_policy"], "saved Spark PipelineModel:", path)
'''),
        (7, "HOLDOUT METRICS + CONFUSION TABLE",
         "The Spark evaluators compute ROC-AUC/PR-AUC; also show positive-class precision, recall, F1, "
         "accuracy and all four confusion cells for **the same Nov–Dec rows**. Always-positive "
         "accuracy and PR-AUC baseline should be compared to *measured* test prevalence.",
         '''\
import pandas as pd
from IPython.display import display
rows = []
for m in observed_2["models"]:
    v = m["metrics"]
    rows.append({"model": m["name"], "family": m["family"],
        "PR-AUC": v["auc_pr"], "ROC-AUC": v["auc_roc"],
        "precision": v["positive_precision"], "recall": v["positive_recall"],
        "F1": v["positive_f1"], "accuracy": v["accuracy"],
        "TN": v["confusion"]["tn"], "FP": v["confusion"]["fp"],
        "FN": v["confusion"]["fn"], "TP": v["confusion"]["tp"],
        "CV_s": m["cv_seconds"], "full_fit_s": m["final_fit_seconds"]})
comparison = pd.DataFrame(rows).set_index("model")
display(comparison)
print("Measured no-skill PR baseline on Nov–Dec = positive prevalence:",
      observed_2["models"][0]["metrics"]["base_rate"])
'''),
        (8, "ONE-CELL EP2: FOUR BEST PARAM/TIMES + CONFUSION + ROC/PR",
         "Rebuild a SINGLE composite EP2 image in THIS cell from all four real CV-winning "
         "parameter sets and fit times, four confusion matrices and held-out ROC/PR panels. "
         "Curves use binned plotting coordinates; numerical AUCs use Spark evaluators.",
         '''\
from IPython.display import Image, display
from coursework.evaluation import plot_model_evidence
series = read_json(project_path(cfg, "results_dir") / "task2_curves.json")["curves"]
ep2 = project_path(cfg, "results_dir") / "task2_evidence.png"
plot_model_evidence(observed_2["models"], series, ep2)  # one cell -> one EP2 composite
display(Image(filename=str(ep2)))
print("Numerical ROC/PR AUC values come from Spark, not binned plot coordinates.")
'''),
        (9, "BEST GRID, OCTOBER THRESHOLD + TREE IMPORTANCE",
         "Show *each candidate grid score* as well as the chosen parameters. Global split importance "
         "comes from fitted tree models and is not a causal effect; Task 3's LIME is one **local** "
         "explanation. The winning algorithm is selected from CV PR-AUC with a predeclared "
         "time tie-breaker, not from Nov–Dec results.",
         '''\
import matplotlib.pyplot as plt
for m in observed_2["models"]:
    print("MODEL:", m["name"], "grid:", m["param_grid"],
          "CV candidate PR-AUCs:", m["cv_scores"], "best:", m["best_params"],
          "October threshold:", m["threshold_tuning"])
rf = next(m for m in observed_2["models"] if m["name"] == "RandomForestClassifier")
importance = pd.DataFrame(rf["tree_global_split_importance"])
display(importance)
ax = importance.head(8).sort_values("global_split_importance").plot.barh(
    x="feature", y="global_split_importance", legend=False, figsize=(8, 3.5), color="#315c84")
ax.set(title="Fitted RF global split importance (not causal)", xlabel="relative importance")
plt.tight_layout(); plt.show()
print("Predeclared CV-selected model:", observed_2["recommended_model"])
'''),
        (10, "MODEL SERIALIZATION (GENUINE SPARK PIPELINEMODEL)",
         "The reference's 'model serialization' cell trains an unrelated small sklearn model and "
         "never saves anything. Here the *real fitted Spark PipelineModel* from Step 6 is reloaded; "
         "the pipeline contains preprocessing and the classifier. Do not recalculate/test-tune "
         "metrics on the final holdout after inspecting them.",
         '''\
from pyspark.ml import PipelineModel
best = observed_2["recommended_model"]
model_path = project_path(cfg, "artifact_dir") / "models" / best
reloaded = PipelineModel.load(str(model_path))
print("Reloaded trained Spark model:", best)
print("Saved stages:", [type(stage).__name__ for stage in reloaded.stages])
print("Measured Task 2 JSON:", project_path(cfg, "results_dir") / "task2.json")
print("Your interpretation must include errors, choice rationale and serving limitations.")
'''),
    ],
    "**Task 2 completion:** document actual four tuned classifiers, their grids/metrics/fit times and "
    "same-cohort comparison. Do not claim a PR-AUC difference is statistically significant "
    "without uncertainty analysis; do not claim booking-time predictions for unknown future payment method.",
)


build(
    "Task3.ipynb", "Task 3 — actual Spark optimisation, stability and explanations",
    "Run after the four real Task 2 models. Show cluster settings, live Spark UI evidence, repeated "
    "before/after experiments, four-model perturbation results, one Spark-backed LIME explanation and "
    "a careful borough proxy audit. Never paste the road-collision notebook's timings or screenshots.",
    [
        (1, "SPARK SESSION + LIVE UI LINK",
         "Capture your **actual** Jobs/Stages timeline and task metrics while this application runs. "
         "`spark.uiWebUrl` can be null if your managed cluster exposes the UI elsewhere; ask your admin "
         "for the correct application link and capture an authentic screenshot.",
         setup("Task3") + '''\
print("Spark UI for THIS application:", spark.sparkContext.uiWebUrl)
'''),
        (2, "READ REAL MODELS + PROCESSED DATA",
         "Verify four observed CV results and saved pipelines before rerunning expensive analysis. "
         "The Task 3 benchmarks operate on the Task 1 Parquet, not a 50k-row sklearn extract.",
         '''\
from coursework.data import load_processed
task2 = read_json(project_path(cfg, "results_dir") / "task2.json")
assert task2["status"] == "observed" and len(task2["models"]) == 4
processed = load_processed(spark, cfg)
print("Four fitted Spark models:", [m["name"] for m in task2["models"]])
print("CV-selected model:", task2["recommended_model"])
'''),
        (3, "MEASURE SHUFFLE, CACHE, SCALABILITY + FOUR-MODEL STABILITY",
         "The code times **identical groupBy actions** under two shuffle settings, separately logs "
         "uncached/cached repeats **and cache-fill cost**, runs 10/25/50/100%-fraction grouped "
         "queries, and applies the SAME frozen +1-hour/masked-borough perturbation to Nov–Dec "
         "TEST inputs for ALL FOUR saved classifiers. Extra retraining trials on Jan–Sep are "
         "separately reported, not substituted for the official test-data perturbation. "
         "This step also computes Spark-backed LIME and borough audit.",
         '''\
from pyspark.storagelevel import StorageLevel
from coursework.task3 import (benchmark_optimisations, benchmark_scalability,
    test_perturbation_analysis, stability_analysis, explain_with_lime,
    fairness_by_borough, run_task3)
print("Benchmark uses:", StorageLevel.MEMORY_AND_DISK)
print("Keep the Spark UI open while this assessed computation runs.")
observed_3 = run_task3(spark, cfg)
assert observed_3["status"] == "observed"
print("Measured optimisation arms:", len(observed_3["optimisations"]))
print("Measured stability models:", list(observed_3["stability"]))
'''),
        (4, "ACTUAL OPTIMISATION RESULTS (NOT A CLAIMED SPEEDUP)",
         "Compare identical actions under different shuffle partitions; cache **fill plus reuse** "
         "may be slower overall. Setting AQE skew-join alone does not prove a groupBy skew was "
         "removed. Record the hardware, spill and stage ID yourself.",
         '''\
import pandas as pd
from IPython.display import display
bench_rows = []
for arm in observed_3["optimisations"]:
    if arm["change"] == "shuffle_partition_count":
        bench_rows.append({"experiment": "shuffle", "setting": arm["partitions"],
                           "repeat_median_s": arm["median_seconds"]})
    else:
        bench_rows.extend([
            {"experiment": "cache", "setting": "uncached repeats",
             "repeat_median_s": arm["median_before_seconds"],
             "total_s": arm["uncached_total_seconds"]},
            {"experiment": "cache", "setting": "cached repeats",
             "repeat_median_s": arm["median_after_seconds"],
             "fill_s": arm["cache_fill_seconds"],
             "total_s": arm["cache_total_seconds_including_fill"]},
        ])
display(pd.DataFrame(bench_rows))
print("Do NOT assert a benefit unless measured total and repeat costs support one.")
'''),
        (5, "SCALABILITY QUERY CURVE (WITH I/O LIMITATION)",
         "10/25/50/100% **sampled Jan–Sep grouped-query** results; even small samples scan "
         "the source Parquet, so the curve is not linear hardware scaling or model-fit scaling.",
         '''\
import matplotlib.pyplot as plt
scaling = pd.DataFrame(observed_3["scalability_query_benchmarks"])
display(scaling[["sample_fraction", "rows", "count_seconds", "group_by_seconds"]])
ax = scaling.plot(x="rows", y="group_by_seconds", marker="o", legend=False,
                  figsize=(7, 3.5), color="#168a85")
ax.set(title="Measured Spark groupBy wall time — training-data fractions",
       xlabel="Sampled rows (input Parquet still scanned)", ylabel="GroupBy seconds")
plt.tight_layout(); plt.show()
'''),
        (6, "PERTURB NOV–DEC TEST INPUTS — FOUR-MODEL ΔF1/ΔROC-AUC",
         "The attached official answer sheet requires a **test-data perturbation**, not "
         "only retraining sensitivity. Read the exact shared Nov–Dec perturbation protocol "
         "and F1/ROC-AUC delta and rank for each model. Labels and October-selected "
         "thresholds stay fixed; Nov–Dec is still NOT used for selecting a new model.",
         '''\
print("Frozen held-out perturbation:", observed_3["test_perturbation_protocol"])
stability_rows = []
for model_name, info in observed_3["stability"].items():
    stability_rows.append({"model": model_name,
        "baseline_F1": info["baseline_metrics"]["positive_f1"],
        "perturbed_F1": info["perturbed_metrics"]["positive_f1"],
        "delta_F1": info["signed_deltas"]["positive_f1"],
        "delta_ROC_AUC": info["signed_deltas"]["auc_roc"],
        "rank": info["rank"]})
stability_table = pd.DataFrame(stability_rows).sort_values("rank")
display(stability_table)
print("Most/least robust under THIS predeclared input-error scenario:",
      observed_3["most_stable"], observed_3["least_stable"])
if observed_3["retraining_stability_supplement"]:
    print("Supplementary Jan-Sep retraining sensitivity (DIFFERENT question):")
    display(pd.DataFrame([{"model": name,
        "mean_abs_october_PR_delta": data["mean_abs_auc_pr_delta"]}
        for name, data in observed_3["retraining_stability_supplement"].items()]))
else:
    print("Extra retraining sensitivity disabled (not required; 16 additional fits).")
'''),
        (7, "LIME ON A SAVED SPARK MODEL (ONE TRIP)",
         "One genuinely fitted Spark PipelineModel scores LIME perturbations; the displayed "
         "weights explain one case, NOT global causality. If the local surrogate has weak "
         "fidelity R², say so rather than claiming a reliable explanation.",
         '''\
from IPython.display import Image, display
lime = observed_3["explainability"]
print("Model and method:", lime["explained_model"], lime["method"])
print("Neighbour queries / local fidelity R²:",
      lime["neighbour_queries"], lime["local_surrogate_fidelity_r2"])
display(pd.DataFrame(lime["feature_weights"]))
display(Image(filename=str(project_path(cfg, "results_dir") / "task3_lime.png")))
'''),
        (8, "BOROUGH AUDIT + BIAS LIMITATION",
         "Report observed prevalence AND prediction rate, FPR and TPR for sufficiently large "
         "borough groups. Geography is a socioeconomic proxy, **not** a protected characteristic "
         "or proof of unlawful disparate impact; card-only tips have selection bias.",
         '''\
groups = pd.DataFrame(observed_3["fairness"]["groups"])
print("Minimum published group size:", observed_3["fairness"]["minimum_group_size"])
print("Suppressed groups:", observed_3["fairness"]["small_groups_suppressed"])
print("Named risk:", observed_3["named_bias_risk"])
display(groups)
if not groups.empty:
    ax = groups.set_index("borough")[["observed_positive_rate",
        "predicted_positive_rate"]].plot.bar(figsize=(8, 3.5), color=["#315c84", "#dc784d"])
    ax.set(ylabel="Rate", title="Observed vs predicted positive rates — borough proxy audit")
    plt.tight_layout(); plt.show()
'''),
        (9, "SPARK UI SCREENSHOT + YOUR DIAGNOSIS",
         "Save a **real** stage screenshot as `results/task3_spark_ui.png` (do not draw or "
         "generate one). In your own report document application/stage ID, slowest vs median "
         "task, shuffle read/write, spill, bottleneck, and whether your measured optimisation "
         "actually helped. The screenshot gate below warns if evidence is absent.",
         '''\
ui_image = project_path(cfg, "results_dir") / "task3_spark_ui.png"
if ui_image.is_file():
    display(Image(filename=str(ui_image)))
else:
    print("MISSING: capture YOUR university Spark UI Stage screenshot at", ui_image)
print("Saved measured Task 3 JSON:", project_path(cfg, "results_dir") / "task3.json")
print("Write YOUR stage/skew/cost explanation. Do not infer a speedup from toggling a flag.")
'''),
        (10, "TASK 3 RESULT + METHODOLOGICAL REFLECTION",
         "Use your measured timings, four stability results, LIME fidelity and suppressed "
         "borough rates to write a critical account. Explain why a cached repeat may appear "
         "fast yet increase total elapsed time, why groupBy sampling isn't hardware scaling, "
         "and why borough/payments limit a fairness claim. This is **not** a generated essay.",
         '''\
print("Task 3 observed status:", observed_3["status"])
print("Saved real Spark UI screenshot:", observed_3["spark_ui_screenshot"])
print("Testable mitigation (not already proven):", observed_3["fairness_mitigation_to_test"])
print("LIME caveat:", observed_3["explainability"]["limitation"])
print("Report your own observations with stage IDs, measured deltas and limitations.")
'''),
    ],
    "**Task 3 completion:** manually inspect genuine Spark UI evidence, interpret measured "
    "trade-offs and cash-tip selection limitations, commit your observed Task 3 work, "
    "and proceed to the four real Tableau dashboards.",
)


build(
    "Task4.ipynb", "Task 4 — four Tableau dashboards and publication evidence",
    "**Run after Tasks 1–3 produce observed results.** Generate small, suppressed **aggregate** "
    "CSV inputs, inspect them and publish one genuine four-dashboard Tableau Public workbook. "
    "The notebook does not claim to create the workbook or fabricate its screenshots.",
    [
        (1, "SPARK SESSION + MEASURED PREREQUISITE CHECK",
         "Use your verified allocation and cluster; read only the prior **observed** task "
         "results before exporting anything to Tableau Public.",
         setup("Task4") + '''\
for number in (1, 2, 3):
    prior = read_json(project_path(cfg, "results_dir") / f"task{number}.json")
    assert prior["status"] == "observed", f"Task {number} is not complete"
print("Tasks 1–3 have real observed result JSONs.")
'''),
        (2, "SPARK AGGREGATIONS + TABLEAU-SAFE EXPORTS",
         "`run_task4` does groupBy over the *real* staged Spark Parquet and writes "
         "quality, model, business and scalability CSVs; borough-hour cells below the "
         "configured threshold are suppressed. First run is normally `pending_publication_or_screenshots` "
         "until YOU make the actual Tableau Public workbook.",
         '''\
from coursework.task4 import run_task4, DASHBOARDS
observed_4 = run_task4(spark, cfg)
for i, name in enumerate(DASHBOARDS, 1):
    print(f"Dashboard {i}: {name}")
print("Exports from actual task results and Spark aggregates:", observed_4["exports"])
print("Publication status (pending until verified):", observed_4["status"])
'''),
        (3, "CSV FILE MANIFEST + SMALL AGGREGATE PREVIEW",
         "Inspect exported field names and the FIRST aggregated row only; no trip-level "
         "records or model predictions should be imported into a PUBLIC Tableau workbook. "
         "Review small-cell suppression across filter combinations before publishing.",
         '''\
import csv
for name in observed_4["exports"]:
    path = project_path(cfg, "results_dir") / name
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header, first_aggregate = reader.fieldnames, next(reader, None)
    print(name, "bytes:", path.stat().st_size,
          "columns:", header, "FIRST AGGREGATE:", first_aggregate)
'''),
        (4, "DASHBOARD 1 — DATA QUALITY + PIPELINE",
         "In Tableau, show actual rows retained, source GiB, partitions, monthly counts and "
         "separately presented non-additive quality flags. This cell previews the real "
         "aggregation in a notebook; it is **not** a substitute for the Tableau dashboard.",
         '''\
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
quality = pd.read_csv(project_path(cfg, "results_dir") / "tbl_d1_quality.csv")
monthly = quality[(quality["section"] == "month") & (quality["metric"] == "clean_rows")]
monthly = monthly.sort_values("key")
display(monthly)
ax = monthly.plot.bar(x="key", y="value", legend=False, color="#315c84", figsize=(8, 3))
ax.set(title="Clean card-only trips per pickup month", xlabel="2019 month", ylabel="Trips")
plt.tight_layout(); plt.show()
'''),
        (5, "DASHBOARD 2 — FOUR-MODEL PERFORMANCE + IMPORTANCE",
         "Show four aligned PR-AUC, ROC-AUC, F1 and CV/full-refit time comparisons, "
         "confusion cells, ROC/PR curves, separate **global** tree importance and **local** "
         "one-case LIME. All held-out metrics use the Nov–Dec cohort.",
         '''\
performance = pd.read_csv(project_path(cfg, "results_dir") / "tbl_d2_performance.csv")
importance = pd.read_csv(project_path(cfg, "results_dir") / "tbl_d2_global_tree_importance.csv")
display(performance[["model", "family", "auc_pr", "auc_roc", "positive_f1",
                     "cv_seconds", "refit_seconds"]])
display(importance.head(10))
ax = performance.plot.bar(x="model", y=["auc_pr", "auc_roc"],
                          figsize=(9, 3.5), color=["#168a85", "#315c84"])
ax.set(title="Measured Nov–Dec AUC by model", xlabel="Classifier", ylabel="AUC")
plt.tight_layout(); plt.show()
'''),
        (6, "DASHBOARD 3 — BUSINESS INSIGHTS (DESCRIPTIVE ONLY)",
         "Explore borough × pickup hour high-tip recorded rates for one month plus "
         "retrospective distance and observed/predicted borough rates; model inputs still "
         "exclude trip distance. Never claim that borough differences are causal or that "
         "an unknown future cash trip has an observed electronic tip.",
         '''\
business = pd.read_csv(project_path(cfg, "results_dir") / "tbl_d3_business.csv")
chosen_month = sorted(business["month"].unique())[-1]
shown = business[(business["month"] == chosen_month) &
                 (business["n"] >= int(cfg["model"]["minimum_borough_size"]))]
print("One-month aggregated preview:", chosen_month)
display(shown.sort_values("n", ascending=False).head(8))
heat = shown.pivot(index="borough", columns="pickup_hour", values="high_tip_rate")
fig, ax = plt.subplots(figsize=(10, 3.5))
# NaNs remain blank for suppressed cells; never colour them as zero rates.
img = ax.imshow(heat.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
ax.set_yticks(range(len(heat.index)), heat.index)
ax.set_xticks(range(len(heat.columns)), heat.columns)
ax.set(xlabel="Pickup hour", ylabel="Borough",
       title=f"Recorded high-tip rate — {chosen_month} (blank cells suppressed)")
fig.colorbar(img, ax=ax, label="Rate")
plt.tight_layout(); plt.show()
'''),
        (7, "DASHBOARD 4 — SCALABILITY + OPTIONAL EVIDENCED COST",
         "Display measured training, shuffle-count, cache and groupBy sample-fraction times. "
         "Price estimates appear only if you supply a verifiable hourly cluster price "
         "and source; never present a shuffle benchmark as hardware scaling.",
         '''\
scale = pd.read_csv(project_path(cfg, "results_dir") / "tbl_d4_scalability.csv")
display(scale[["experiment", "model", "config", "wall_seconds", "usd_estimate"]].head(16))
train_times = scale[scale["experiment"] == "training"]
ax = train_times.plot.bar(x="model", y="wall_seconds", legend=False,
                          figsize=(8, 3), color="#dc784d")
ax.set(title="Actual CV + refit duration", xlabel="Spark classifier", ylabel="Seconds")
plt.tight_layout(); plt.show()
print("Cost source:", cfg["tableau"].get("cluster_cost_source") or "No price entered")
'''),
        (8, "BUILD ONE FOUR-DASHBOARD TABLEAU PUBLIC WORKBOOK",
         "Use `tableau/BUILD_DASHBOARDS.md` as a construction guide. Publish **your own** workbook "
         "with the four named dashboards above, open the public URL while signed out, and check "
         "each view. Export four genuine screenshots to `results/tableau_dashboard_1.png` "
         "through `_4.png`. Enter the verified public URL and confirmation flag into your "
         "gitignored config. A Python HTTP check cannot prove four dashboards exist.",
         '''\
from coursework.task4 import verify_public_link
url = cfg["tableau"].get("public_workbook_url", "")
print("Actual live Tableau Public URL check:", verify_public_link(url))
print("Real four-dashboard screenshot inventory:",
      [(i, (project_path(cfg, "results_dir") / f"tableau_dashboard_{i}.png").is_file())
       for i in range(1, 5)])
print("Current publication status:", observed_4["status"])
'''),
        (9, "REVERIFY PUBLICATION + REAL FOUR-PANEL CONTACT SHEET",
         "After you publish and capture the four genuine dashboards, reload the local "
         "config in case you edited it since Step 1, then rerun Task 4 to verify the URL "
         "and assemble the **real screenshots**. Do not substitute generated mock images. "
         "A pending status is expected if publication is unfinished.",
         '''\
from IPython.display import Image, display
cfg = load_config()
images_ready = all((project_path(cfg, "results_dir") /
                    f"tableau_dashboard_{i}.png").is_file() for i in range(1, 5))
if cfg["tableau"].get("published_four_dashboards_verified") and images_ready:
    observed_4 = run_task4(spark, cfg)
else:
    print("Publication not complete: retain pending status; no contact sheet invented.")
print("Task 4 status:", observed_4["status"], observed_4["link_check"])
sheet = project_path(cfg, "results_dir") / "task4_dashboard_contact_sheet.png"
if sheet.is_file() and observed_4["status"] == "observed":
    display(Image(filename=str(sheet)))
else:
    print("No genuine verified four-dashboard contact sheet yet.")
'''),
        (10, "THREE INSIGHTS + FINAL SUBMISSION CHECK",
         "Write three **your-own-words** business observations, each with a real chart/number "
         "and an appropriate decision and uncertainty. Capture meaningful private-org commits, "
         "a reviewed Word report, four runnable notebooks and an honest AI-use declaration. "
         "The checker cannot certify a grade or the authenticity of evidence.",
         '''\
print("Four dashboard themes:", DASHBOARDS)
print("Verified workbook URL:", observed_4["tableau_public_link"])
print("Observed completion status (must be 'observed' before submission):", observed_4["status"])
print("Write each insight as: observation + specific measured figure + decision + caveat.")
print("Then run: python scripts/check_submission.py --report path/to/reviewed.docx")
'''),
    ],
    "**Task 4 completion:** the four Tableau Public dashboards, URL, screenshots, "
    "contact sheet, report and commits must be genuine student work. An unexecuted "
    "notebook or a workbook construction guide does not constitute publication.",
)
