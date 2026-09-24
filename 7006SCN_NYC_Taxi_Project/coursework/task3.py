"""Measured Spark optimisation, retraining perturbation, local LIME, fairness audit."""
from __future__ import annotations

import math
import statistics
import time
from pathlib import Path

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession, functions as F
from pyspark.storagelevel import StorageLevel

from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, load_processed, split_processed
from .evaluation import apply_threshold, full_metrics, scored_frame
from .models import build_model_specs, pipeline_for, use_best_params
from .settings import now_utc, project_path, read_json, require_verified_allocation, spark_configuration, write_json
from .task2 import training_weights


def _group_action(frame) -> int:
    # Same action in each benchmark arm. Collect only ~6 borough x 24 hour groups.
    groups = frame.groupBy("pickup_borough", "pickup_hour").agg(
        F.count(F.lit(1)).alias("trips"), F.avg("label").alias("observed_rate")
    ).collect()
    return len(groups)


def benchmark_optimisations(spark: SparkSession, data, *, repeat: int = 2) -> list[dict]:
    """After a common warmup, compare repeated identical actions; do not claim
    'speedup' when re-partitioning or caching does not actually improve time.
    Cache fill is timed separately; hardware/OS effects are not randomised.
    """
    train, _, _ = split_processed(data)
    train.count()   # warm OS and Spark read path, not included in timed comparison
    original = spark.conf.get("spark.sql.shuffle.partitions")
    observations = []
    try:
        for n in [int(original), max(2, int(original) // 2)]:
            spark.conf.set("spark.sql.shuffle.partitions", n)
            runs = []
            for _ in range(repeat):
                start = time.perf_counter()
                groups = _group_action(train)
                runs.append(time.perf_counter() - start)
            observations.append({"change": "shuffle_partition_count", "partitions": n,
                                 "group_count": groups, "seconds": runs, "median_seconds": statistics.median(runs)})
    finally:
        spark.conf.set("spark.sql.shuffle.partitions", original)
    uncached = []
    for _ in range(repeat):
        start = time.perf_counter()
        group_count = _group_action(train)
        uncached.append(time.perf_counter() - start)
    cached = train.persist(StorageLevel.MEMORY_AND_DISK)
    try:
        start = time.perf_counter()
        materialised = cached.count()
        fill_seconds = time.perf_counter() - start
        cached_runs = []
        for _ in range(repeat):
            start = time.perf_counter()
            _group_action(cached)
            cached_runs.append(time.perf_counter() - start)
    finally:
        cached.unpersist(blocking=True)
    observations.append({"change": "persist_MEMORY_AND_DISK", "rows_materialised": materialised,
        "groups": group_count, "uncached_seconds": uncached, "cached_seconds": cached_runs,
        "median_before_seconds": statistics.median(uncached),
        "median_after_seconds": statistics.median(cached_runs),
        "cache_fill_seconds": fill_seconds,
        "uncached_total_seconds": sum(uncached),
        "cache_total_seconds_including_fill": fill_seconds + sum(cached_runs),
        "break_even_note": "Caching only pays if enough actions reuse the same persisted DataFrame; compare total time incl. fill."})
    return observations


def benchmark_scalability(data, seed: int, fractions: tuple[float, ...] = (0.10, 0.25, 0.50, 1.0)) -> list[dict]:
    """Measure identical grouped action for sampled training fractions.

    Spark still scans the input Parquet for sample(False, p); wall time therefore
    includes a near-constant source-read cost. This is an empirical query-scaling
    curve, NOT a claim about 10-to-100% model-training speedup.
    """
    train, _, _ = split_processed(data)
    output = []
    for fraction in fractions:
        sub = train.sample(False, fraction, seed=seed)
        start = time.perf_counter()
        n = sub.count()
        count_seconds = time.perf_counter() - start
        start = time.perf_counter()
        groups = _group_action(sub)
        group_seconds = time.perf_counter() - start
        output.append({"sample_fraction": fraction, "rows": n, "count_seconds": round(count_seconds, 3),
            "group_by_seconds": round(group_seconds, 3), "group_count": groups,
            "interpretation_limit": "Spark reads source files to sample; times include IO; this is groupBy query scaling, not ML training scaling."})
    return output


def perturb_test_features(test, *, seed: int, fraction: float = 0.20):
    """Fixed TEST-data feature perturbation for the answer sheet's Task 3.2.

    Half the designated rows have pickup hour shifted by +1; the other half
    have pickup borough replaced by Unknown. Labels, time split, payment cohort
    and outcome fields are NEVER changed. Every model sees the SAME materialised
    perturbed Nov--Dec holdout. This is a post-selection robustness check, not
    another opportunity to tune the four models or their October thresholds.
    """
    if not 0 < fraction < 1:
        raise ValueError("test_perturbation_fraction must be strictly between zero and one")
    draw = F.col("_stability_draw")
    hour = draw < F.lit(fraction / 2)
    borough = (draw >= F.lit(fraction / 2)) & (draw < F.lit(fraction))
    return (test.withColumn("_stability_draw", F.rand(seed))
        .withColumn("_stability_hour_changed", hour)
        .withColumn("_stability_borough_changed", borough & (F.col("pickup_borough") != "Unknown"))
        .withColumn("pickup_hour", F.when(hour,
            F.pmod(F.col("pickup_hour").cast("int") + F.lit(1), F.lit(24)).cast("string"))
            .otherwise(F.col("pickup_hour")))
        .withColumn("pickup_borough", F.when(borough, F.lit("Unknown"))
            .otherwise(F.col("pickup_borough")))
        .drop("_stability_draw"))


def test_perturbation_analysis(spark: SparkSession, cfg: dict,
                               model_results: list[dict], data) -> dict:
    """Measure held-out F1 and ROC-AUC changes for ALL FOUR fitted pipelines.

    Perturbed rows are written ONCE to shared Parquet, freezing the random draw
    before separate model actions. Existing Task2 metrics refer to the identical
    unperturbed holdout; the October decision thresholds stay fixed. Per-model
    score output is materialised before multiple Spark evaluator actions.
    """
    _, _, test = split_processed(data)
    if len(model_results) != 4:
        raise ValueError("Four real Task2 models are needed for the test perturbation")
    fraction = float(cfg["model"].get("test_perturbation_fraction", 0.20))
    seed = int(cfg["model"]["seed"]) + 301
    base_rows = {int(row["metrics"]["rows"]) for row in model_results}
    if len(base_rows) != 1:
        raise ValueError("Task2 models did not score the same original Nov-Dec holdout")
    destination = project_path(cfg, "artifact_dir") / "stability_test_perturbed"
    perturb_test_features(test, seed=seed, fraction=fraction).write.mode("overwrite").parquet(str(destination))
    fixed_test = spark.read.parquet(str(destination))
    counts = fixed_test.agg(
        F.count(F.lit(1)).alias("rows"),
        F.sum("label").alias("positive_labels"),
        F.sum(F.col("_stability_hour_changed").cast("long")).alias("hour_shifted"),
        F.sum(F.col("_stability_borough_changed").cast("long")).alias("borough_hidden"),
    ).first().asDict()
    n = int(counts["rows"])
    if n != next(iter(base_rows)) or n == 0:
        raise ValueError("Perturbed holdout has a different row count from the Task2 test")
    changed_labels = int(counts["positive_labels"] or 0)
    if any(round(float(row["metrics"]["base_rate"]) * n) != changed_labels
           for row in model_results):
        raise ValueError("Task2's holdout label distribution differs from the perturbed test cohort")
    changes = int(counts["hour_shifted"] or 0) + int(counts["borough_hidden"] or 0)
    if changes <= 0:
        raise ValueError("Perturbation changed no pickup-time features; check the test cohort")
    output: dict[str, dict] = {}
    for row in model_results:
        name = row["name"]
        model_path = project_path(cfg, "artifact_dir") / "models" / name
        model = PipelineModel.load(str(model_path))
        scored_path = project_path(cfg, "artifact_dir") / "stability_predictions" / name
        scored_frame(model, fixed_test, name).write.mode("overwrite").parquet(str(scored_path))
        changed = spark.read.parquet(str(scored_path))
        threshold = float(row["threshold_tuning"]["threshold"])
        after = full_metrics(changed, threshold)
        before = row["metrics"]
        if int(after["rows"]) != n:
            raise ValueError(f"{name} scored a different perturbed test population")
        signed = {key: float(after[key] - before[key]) for key in
                  ("positive_f1", "auc_roc", "auc_pr")}
        output[name] = {
            "baseline_metrics": {key: float(before[key]) for key in signed},
            "perturbed_metrics": {key: float(after[key]) for key in signed},
            "signed_deltas": signed,
            "impact_score": abs(signed["positive_f1"]) + abs(signed["auc_roc"]),
            "fixed_october_threshold": threshold,
        }
        print(f"{name}: held-out test ΔF1={signed['positive_f1']:+.5f}; "
              f"ΔROC-AUC={signed['auc_roc']:+.5f}", flush=True)
    ordered = sorted(output, key=lambda name: (output[name]["impact_score"], name))
    for rank, name in enumerate(ordered, 1):
        output[name]["rank"] = rank
    return {
        "protocol": {
            "cohort": "same Nov-Dec 2019 test rows as Task2, measured once after model selection",
            "fraction_requested": fraction, "seed": seed,
            "perturbation": "half +1 pickup hour; half pickup borough hidden as Unknown; disjoint masks",
            "rows": n, "positive_labels": changed_labels,
            "hour_shifted": int(counts["hour_shifted"] or 0),
            "borough_hidden": int(counts["borough_hidden"] or 0),
            "labels_changed": 0,  # verified against Task2 row counts + positives
            "same_test_rows_as_task2": True,
            "rank_rule": "smallest |ΔF1| + |ΔROC-AUC|; tied models sorted by name",
            "interpretation_limit": "One fixed synthetic input-error scenario is not evidence of real-world drift, fairness or model safety.",
        },
        "models": output, "most_stable": ordered[0], "least_stable": ordered[-1],
    }


def stability_analysis(spark: SparkSession, cfg: dict, model_results: list[dict], data) -> dict:
    params = cfg["model"]
    train, val, _ = split_processed(data)
    train_w, _ = training_weights(train)
    fraction = float(params["stability_fraction"])
    subsample = float(params["stability_subsample_fraction"])
    trials = int(params["stability_trials"])
    if not 0 < fraction <= 1 or not 0 < subsample < 1 or trials < 2:
        raise ValueError("Set 0<stability_fraction<=1, 0<subsample<1 and stability_trials>=2")
    base = train_w.sample(False, fraction, seed=int(params["seed"]) + 91).persist(StorageLevel.MEMORY_AND_DISK)
    # Fixed evaluation subset throughout; no score drift from a changing test population.
    fixed_val = val.sample(False, fraction, seed=int(params["seed"]) + 92).persist(StorageLevel.MEMORY_AND_DISK)
    output = {}
    try:
        nbase, nval = base.count(), fixed_val.count()
        if nbase < 100 or nval < 100:
            raise ValueError("Stability samples too small; increase stability_fraction")
        specs = {spec.name: spec for spec in build_model_specs()}
        for row in model_results:
            spec = specs[row["name"]]
            estimator = use_best_params(spec, row["best_params"])
            threshold = float(row["threshold_tuning"]["threshold"])
            baseline_model = pipeline_for(estimator).fit(base)
            baseline = full_metrics(scored_frame(baseline_model, fixed_val, spec.name), threshold)
            iterations = []
            for trial in range(trials):
                perturbed = base.sample(False, subsample, seed=int(params["seed"]) + 200 + trial)
                ntrial = perturbed.count()
                if ntrial < 100:
                    raise ValueError(f"Trial {trial} too small; adjust stability_fraction")
                t0 = time.perf_counter()
                model = pipeline_for(estimator).fit(perturbed)
                fit_time = time.perf_counter() - t0
                measured = full_metrics(scored_frame(model, fixed_val, spec.name), threshold)
                deltas = {key: measured[key] - baseline[key] for key in ("auc_pr", "auc_roc", "positive_f1")}
                iterations.append({"trial": trial + 1, "train_rows": ntrial, "fit_seconds": round(fit_time, 3),
                                   "metrics": {k: measured[k] for k in deltas}, "signed_deltas": deltas})
            output[spec.name] = {
                "baseline_sample_train_rows": nbase, "fixed_oct_validation_rows": nval,
                "baseline_metrics": {k: baseline[k] for k in ("auc_pr", "auc_roc", "positive_f1")},
                "perturbation": f"independent {subsample:.1%} WITHOUT-replacement subsamples of a fixed Jan-Sep sample; NOT a bootstrap",
                "trials": iterations,
                "mean_abs_auc_pr_delta": statistics.mean(abs(x["signed_deltas"]["auc_pr"]) for x in iterations),
                "max_abs_auc_pr_delta": max(abs(x["signed_deltas"]["auc_pr"]) for x in iterations),
            }
            print(f"{spec.name}: max |delta PR-AUC|={output[spec.name]['max_abs_auc_pr_delta']:.5f}")
        return output
    finally:
        base.unpersist(blocking=True)
        fixed_val.unpersist(blocking=True)


def fairness_by_borough(scored, threshold: float, minimum_size: int = 1000) -> dict:
    """Borough is a geographic PROXY, not a legal protected class. No causal claims."""
    frame = apply_threshold(scored, threshold).where(F.col("pickup_borough") != "Unknown")
    grouped = frame.groupBy("pickup_borough").agg(
        F.count(F.lit(1)).alias("n"),
        F.sum("label").alias("observed_positive"),
        F.sum("decision").alias("predicted_positive"),
        F.sum(F.when((F.col("label") == 1) & (F.col("decision") == 1), 1).otherwise(0)).alias("tp"),
        F.sum(F.when((F.col("label") == 0) & (F.col("decision") == 1), 1).otherwise(0)).alias("fp"),
    ).collect()
    small, reported = [], []
    for x in grouped:
        d = x.asDict()
        if d["n"] < minimum_size:
            small.append({"borough": d["pickup_borough"], "n": d["n"]})
            continue
        pos, predicted, tp, fp, total = (int(d[k] or 0) for k in
            ("observed_positive", "predicted_positive", "tp", "fp", "n"))
        reported.append({"borough": d["pickup_borough"], "n": total,
            "observed_positive_rate": pos / total, "predicted_positive_rate": predicted / total,
            "tpr": tp / pos if pos else None,
            "fpr": fp / (total - pos) if total - pos else None,
            "precision": tp / predicted if predicted else None})
    rates = [row["predicted_positive_rate"] for row in reported]
    ratio = min(rates) / max(rates) if rates and max(rates) > 0 else None
    return {"min_over_max_predicted_positive_rate": ratio, "groups": reported,
            "small_groups_suppressed": small, "minimum_group_size": minimum_size,
            "interpretation_limit": "Boroughs are geography, not protected attributes; disparities may reflect base rates, label under-reporting or sampling, not established discrimination."}


def explain_with_lime(spark: SparkSession, cfg: dict, data, model_name: str) -> dict:
    """LIME queries the *actual saved Spark PipelineModel* on <=N synthetic local neighbours.

    Only a capped background and one explained case enter the driver; Spark scores
    all LIME neighbours as a single small batch. Local weights are NOT global SHAP.
    """
    from lime.lime_tabular import LimeTabularExplainer
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

    # Explanations perturb RAW month as one categorical input. The Spark scorer
    # re-derives sin/cos together, so LIME never samples an off-circle month pair.
    lime_categories = [*CATEGORICAL_FEATURES, "pickup_month"]
    features = ["passenger_count", *lime_categories]
    train, _, test = split_processed(data)
    limit = int(cfg["model"]["lime_background_rows"])
    if limit < 20:
        raise ValueError("LIME needs >=20 background examples")
    seed = int(cfg["model"]["seed"])
    sampling = float(cfg["model"].get("lime_background_fraction", 0.0005))
    if not 0 < sampling <= 1:
        raise ValueError("lime_background_fraction must be in (0, 1]")
    background = (train.select(*features).sample(False, sampling, seed).orderBy(F.rand(seed))
                  .limit(limit).collect())
    if len(background) < 20:
        raise ValueError("Too few LIME background rows; increase sampling fraction or use full data")
    instance = test.select(*features).where(F.col("passenger_count").isNotNull()).limit(1).first()
    if instance is None:
        raise ValueError("No complete final-test row available to explain")
    categories = {}
    for column in lime_categories:
        names = sorted({str(r[column]) for r in background} | {str(instance[column]), "Unknown"})
        categories[column] = {value: i for i, value in enumerate(names)}
    def encode(row):
        return [float(row["passenger_count"] if row["passenger_count"] is not None else 1.0)] + [
            float(categories[col][str(row[col])]) for col in lime_categories]
    array = np.asarray([encode(row) for row in background], dtype=float)
    item = np.asarray(encode(instance), dtype=float)
    decoder = {col: {int(v): k for k, v in lookup.items()} for col, lookup in categories.items()}
    schema = StructType([StructField("_lime_id", LongType(), False),
        StructField("passenger_count", DoubleType(), True)] +
        [StructField(col, StringType(), True) for col in lime_categories])
    model = PipelineModel.load(str(project_path(cfg, "artifact_dir") / "models" / model_name))
    def probability(batch):
        records = []
        for idx, values in enumerate(batch):
            decoded = []
            for position, col in enumerate(lime_categories, start=1):
                value = int(round(values[position]))
                value = max(0, min(value, len(decoder[col]) - 1))
                decoded.append(decoder[col][value])
            records.append((idx, float(values[0]), *decoded))
        to_score = spark.createDataFrame(records, schema=schema)
        angle = F.col("pickup_month").cast("double") * F.lit(2 * math.pi / 12)
        to_score = to_score.withColumn("pickup_month_sin", F.sin(angle)).withColumn("pickup_month_cos", F.cos(angle))
        scores = (model.transform(to_score)
            .select("_lime_id", vector_to_array("probability")[1].alias("p"))
            .orderBy("_lime_id").collect())
        pos = np.asarray([float(row["p"]) for row in scores])
        if len(pos) != len(batch):
            raise RuntimeError("LIME predictions did not align with their input rows")
        return np.column_stack([1 - pos, pos])
    explainer = LimeTabularExplainer(
        array, feature_names=features, class_names=["not high tip", "high tip"], mode="classification",
        categorical_features=list(range(1, len(features))),
        categorical_names={i: [decoder[col][v] for v in range(len(decoder[col]))]
                           for i, col in enumerate(lime_categories, start=1)},
        discretize_continuous=False, random_state=seed,
    )
    explanation = explainer.explain_instance(item, probability, labels=(1,),
                                             num_features=min(7, len(features)),
                                             num_samples=int(cfg["model"]["lime_num_samples"]))
    raw_weights = explanation.as_list(label=1)
    top = [{"feature": feature, "local_weight": float(weight)} for feature, weight in raw_weights]
    fig = explanation.as_pyplot_figure(label=1)
    fig.suptitle(f"LIME: one final-test trip; actual Spark {model_name} predictions")
    path = project_path(cfg, "results_dir") / "task3_lime.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return {"method": "LIME local surrogate on actual Spark model", "explained_model": model_name,
            "background_train_rows": len(array), "neighbour_queries": int(cfg["model"]["lime_num_samples"]),
            "feature_weights": top, "five_or_more_features": len(top) >= 5,
            "local_surrogate_fidelity_r2": float(explanation.score),
            "figure_path": str(path),
            "limitation": "Local approximation for ONE trip; LIME perturbs categorical values and can create implausible combinations. Check surrogate R2."}


def run_task3(spark: SparkSession, cfg: dict) -> dict:
    require_verified_allocation(cfg)
    prior = read_json(project_path(cfg, "results_dir") / "task2.json")
    if prior.get("status") != "observed" or len(prior.get("models", [])) != 4:
        raise ValueError("Run all four real Task2 CVs/refits before Task3")
    data = load_processed(spark, cfg)
    optimisations = benchmark_optimisations(spark, data)
    scalability = benchmark_scalability(data, int(cfg["model"]["seed"]))
    # The attached official answer sheet explicitly requires a perturbation to
    # the TEST DATA for all four models (Task 3.2). This comes first and is the
    # reported stability/ranking; retraining sensitivity is an extra analysis.
    test_stability = test_perturbation_analysis(spark, cfg, prior["models"], data)
    # Extra train-subsample refits are expensive and NOT requested by Task 3.2.
    # Keep the method available, but do not silently add 16 more model fits.
    retraining = (stability_analysis(spark, cfg, prior["models"], data)
                  if cfg["model"].get("run_retraining_stability", False) else {})
    chosen = prior["recommended_model"]
    # All four selected models now expose a Spark probability vector, including
    # the neural third family. Explain the ACTUAL CV-selected model, not a proxy.
    lime = explain_with_lime(spark, cfg, data, chosen)
    _, _, test = split_processed(data)
    selected = next(r for r in prior["models"] if r["name"] == chosen)
    scored = spark.read.parquet(str(project_path(cfg, "artifact_dir") / "predictions" / chosen))
    fairness = fairness_by_borough(scored, selected["threshold_tuning"]["threshold"],
                                   int(cfg["model"]["minimum_borough_size"]))
    screenshot = project_path(cfg, "results_dir") / "task3_spark_ui.png"
    output = {
        "status": "observed", "generated_utc": now_utc(),
        "optimisations": optimisations, "scalability_query_benchmarks": scalability,
        "test_perturbation_protocol": test_stability["protocol"],
        "stability": test_stability["models"],
        "retraining_stability_supplement": retraining,
        "stability_ranking_metric": test_stability["protocol"]["rank_rule"],
        "most_stable": test_stability["most_stable"],
        "least_stable": test_stability["least_stable"],
        "explainability": lime, "named_bias_risk": "Pickup borough can proxy socioeconomic/geographic disadvantage; credit-card-only outcome selection.",
        "fairness": fairness,
        "fairness_mitigation_to_test": "Compare borough-blind model on identical holdout, calibrate by group only with stakeholder review; neither removes missing cash-tip labels.",
        "spark_ui_screenshot": str(screenshot) if screenshot.is_file() else None,
        "spark_ui_manual_interpretation_required": True,
        "spark_configuration": spark_configuration(spark),
    }
    write_json(project_path(cfg, "results_dir") / "task3.json", output)
    return output
