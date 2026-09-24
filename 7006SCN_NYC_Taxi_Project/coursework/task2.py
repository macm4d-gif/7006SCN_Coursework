"""Task 2: four actual Spark ML CrossValidators, refits, holdout metrics."""
from __future__ import annotations

import time
from pathlib import Path

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator
from pyspark.sql import SparkSession, functions as F
from pyspark.storagelevel import StorageLevel

from .data import load_processed, split_processed
from .evaluation import binned_curve, full_metrics, plot_model_evidence, scored_frame, select_threshold
from .models import (FEATURE_VECTOR_SIZE, build_model_specs, param_grid_as_dicts,
                     pipeline_for, tree_feature_importance, use_best_params)
from .settings import now_utc, project_path, read_json, require_verified_allocation, spark_configuration, write_json


def training_weights(train) -> tuple[object, dict]:
    counts = {int(row["label"]): int(row["count"]) for row in train.groupBy("label").count().collect()}
    if 0 not in counts or 1 not in counts:
        raise ValueError(f"Cannot fit binary classification without both target classes: {counts}")
    total = counts[0] + counts[1]
    # Inverse prevalence estimated ONLY on Jan--Sep; use same weights for every family.
    w0, w1 = total / (2 * counts[0]), total / (2 * counts[1])
    weighted = train.withColumn("class_weight", F.when(F.col("label") == 1.0, w1).otherwise(w0))
    return weighted, {"counts": counts, "negative": w0, "positive": w1}


def run_task2(spark: SparkSession, cfg: dict, *, smoke: bool = False) -> dict:
    if not smoke:
        require_verified_allocation(cfg)
        prior = read_json(project_path(cfg, "results_dir") / "task1.json")
        if prior.get("status") != "observed" or not all(prior["big_data_verification"].values()):
            raise ValueError("Task1 must measure full compliant allocated data before Task2")
    params = cfg["model"]
    folds = 2 if smoke else int(params["cv_folds"])
    tune_fraction = 1.0 if smoke else float(params["tuning_fraction"])
    if folds < 2 or not 0 < tune_fraction <= 1:
        raise ValueError("cv_folds>=2 and 0<tuning_fraction<=1 are required")
    data = load_processed(spark, cfg)
    train, validation, test = split_processed(data)
    weighted, balance = training_weights(train)
    tune = weighted.sample(False, tune_fraction, seed=int(params["seed"]))
    # Assign whole calendar days to folds: no journey/day appears in two folds.
    # CrossValidator nevertheless reuses later months for some folds: November-December
    # remains untouched; report this CV limitation rather than claiming forward CV.
    tune = tune.withColumn("cv_fold", F.pmod(F.xxhash64("pickup_date"), F.lit(folds)).cast("int"))
    tune = tune.persist(StorageLevel.MEMORY_AND_DISK)
    tune_counts = {int(x["label"]): int(x["count"]) for x in tune.groupBy("label").count().collect()}
    if 0 not in tune_counts or 1 not in tune_counts:
        tune.unpersist()
        raise ValueError(f"CV sample contains one class only: {tune_counts}")
    results = []
    curves = []
    results_dir = project_path(cfg, "results_dir")
    artifact_dir = project_path(cfg, "artifact_dir")
    try:
        for spec in build_model_specs(smoke=smoke):
            print(f"Training {spec.name}: {folds} folds x {len(spec.param_grid)} candidate grids", flush=True)
            cv = CrossValidator(
                estimator=pipeline_for(spec.estimator),
                estimatorParamMaps=spec.param_grid,
                evaluator=BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR"),
                numFolds=folds, foldCol="cv_fold", parallelism=1 if smoke else int(params["cv_parallelism"]),
                collectSubModels=False, seed=int(params["seed"]),
            )
            started = time.perf_counter()
            cv_fit = cv.fit(tune)
            cv_seconds = time.perf_counter() - started
            best_i = max(range(len(cv_fit.avgMetrics)), key=lambda i: cv_fit.avgMetrics[i])
            selected = {param.name: value for param, value in spec.param_grid[best_i].items()}
            tuned_estimator = use_best_params(spec, selected)
            started = time.perf_counter()
            final = pipeline_for(tuned_estimator).fit(weighted)
            fit_seconds = time.perf_counter() - started
            if not smoke:
                final.write().overwrite().save(str(artifact_dir / "models" / spec.name))
            global_importance = tree_feature_importance(final, test) if spec.name in (
                "RandomForestClassifier", "GBTClassifier") else []
            val_scored = scored_frame(final, validation, spec.name).persist(StorageLevel.MEMORY_AND_DISK)
            try:
                val_scored.count()
                threshold = select_threshold(val_scored, int(params["curve_bins"]))
            finally:
                val_scored.unpersist()
            started = time.perf_counter()
            test_scored = scored_frame(final, test, spec.name)
            if smoke:
                # Unit/integration smoke runs do not write assessable results.
                scored_for_eval = test_scored.persist(StorageLevel.MEMORY_AND_DISK)
                scored_for_eval.count()
            else:
                predictions = artifact_dir / "predictions" / spec.name
                test_scored.write.mode("overwrite").parquet(str(predictions))
                scored_for_eval = spark.read.parquet(str(predictions))
            predict_seconds = time.perf_counter() - started
            started = time.perf_counter()
            metrics = full_metrics(scored_for_eval, threshold["threshold"])
            points, curve_info = binned_curve(scored_for_eval, int(params["curve_bins"]))
            evaluation_seconds = time.perf_counter() - started
            if smoke:
                scored_for_eval.unpersist()
            result = {
                "name": spec.name, "family": spec.family, "selection_reason_to_verify": spec.rationale_to_check,
                "feature_vector_width": FEATURE_VECTOR_SIZE,
                "training_weight_policy": (
                    "inverse Jan-Sep class frequency via Spark weightCol"
                    if spec.estimator.hasParam("weightCol") else
                    "unweighted: Spark MLP has no weightCol; same train/CV rows, different objective; interpret cross-family comparisons accordingly"
                ),
                "param_grid": param_grid_as_dicts(spec), "best_params": selected,
                "cv_folds": folds, "cv_metric": "areaUnderPR", "cv_scores": list(map(float, cv_fit.avgMetrics)),
                "cv_best_auc_pr": float(cv_fit.avgMetrics[best_i]),
                "cv_seconds": round(cv_seconds, 3), "final_fit_seconds": round(fit_seconds, 3),
                "prediction_write_seconds": round(predict_seconds, 3),
                "evaluation_seconds": round(evaluation_seconds, 3),
                "threshold_tuning": threshold, "metrics": metrics,
                "tree_global_split_importance": global_importance,
                "test_curve_bins": curve_info,
                "cluster_configuration": spark_configuration(spark),
            }
            results.append(result)
            curves.append({"model": spec.name, "points": points})
            print(f"{spec.name}: CV {cv_seconds:.1f}s, full refit {fit_seconds:.1f}s, "
                  f"test PR-AUC {metrics['auc_pr']:.4f}, ROC-AUC {metrics['auc_roc']:.4f}", flush=True)
            if not smoke:
                write_json(results_dir / "task2.json", {"status": "partial - do not submit", "models": results,
                        "generated_utc": now_utc(), "expected_models": 4})
        # PREDECLARED recommendation uses the Jan--Sep CV criterion + measured training cost,
        # never the untouched final-test scores to choose the winner.
        top = max(model["cv_best_auc_pr"] for model in results)
        within = [r for r in results if top - r["cv_best_auc_pr"] <= 0.005]
        chosen = min(within, key=lambda r: r["cv_seconds"] + r["final_fit_seconds"])
        output = {
            "status": "smoke_test_only" if smoke else "observed", "generated_utc": now_utc(),
            "pool_reference": cfg["allocation"]["pool_reference"],
            "comparison_protocol": "same Jan-Sep training, identical October threshold period and untouched Nov-Dec test",
            "cv_limit": "Fold day blocks are not forward-chaining; November-December is never used in CV/threshold selection",
            "tuning_sampling_fraction": tune_fraction, "tuning_sample_class_counts": tune_counts,
            "training_class_weights_from_jan_sep": balance,
            "weighting_limit": "Spark MLP has no weightCol; unlike LR/RF/GBT, it fits unweighted on the identical training rows; PR-AUC and per-model October threshold evaluation remain on shared cohorts.",
            "metric_priority": "CV AUC-PR; for gaps <=0.005, prefer shorter CV+full-refit time",
            "threshold_priority": "October F0.5; reduces false positives compared with uncalibrated default",
            "models": results, "recommended_model": chosen["name"],
            "recommendation_reason": "Highest CV PR-AUC; within 0.005 choose shorter measured CV+refit time. Test metrics reported once, not used in model selection.",
        }
        if not smoke:
            plot_model_evidence(results, curves, results_dir / "task2_evidence.png")
            write_json(results_dir / "task2_curves.json", {"status": "observed", "curves": curves})
            write_json(results_dir / "task2.json", output)
        return output
    finally:
        tune.unpersist()
