"""Distributed, comparable binary-classification evaluation.

Only small histogram aggregates and confusion cells leave Spark. Binary AUCs come
from Spark's BinaryClassificationEvaluator, not invented / copied numbers.
"""
from __future__ import annotations

from pathlib import Path

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, functions as F


def scored_frame(model, frame: DataFrame, name: str) -> DataFrame:
    transformed = model.transform(frame)
    score = vector_to_array("probability")[1]  # All four output a score vector; not necessarily calibrated.
    return transformed.select(
        F.col("label").cast("double"), score.cast("double").alias("score"),
        "year_month", "pickup_borough", "pickup_date", "PULocationID", "trip_distance",
    )


def binned_curve(scored: DataFrame, bins: int = 150) -> tuple[list[dict], dict]:
    """Approximate only PLOT coordinates; AUC metrics are evaluated separately."""
    if bins < 2:
        raise ValueError("Need at least two score bins")
    statistics = scored.agg(F.min("score").alias("min_score"), F.max("score").alias("max_score"),
                            F.count(F.lit(1)).alias("rows"), F.sum("label").alias("positives")).first().asDict()
    if not statistics["rows"] or statistics["positives"] in (None, 0, statistics["rows"]):
        raise ValueError(f"Undefined ROC/PR for one-class data: {statistics}")
    low, high = float(statistics["min_score"]), float(statistics["max_score"])
    if high == low:
        bucket = F.lit(0)
    else:
        bucket = F.least(F.lit(bins - 1), F.greatest(F.lit(0),
            F.floor((F.col("score") - F.lit(low)) / F.lit(high - low) * F.lit(bins)).cast("int")))
    small = (
        scored.withColumn("bucket", bucket)
        .groupBy("bucket").agg(F.count(F.lit(1)).alias("n"), F.sum("label").alias("positive"))
        .orderBy(F.desc("bucket")).collect()
    )
    p, n = int(statistics["positives"]), int(statistics["rows"] - statistics["positives"])
    points = [{"threshold": high + 1e-9, "fpr": 0.0, "tpr": 0.0,
               "recall": 0.0, "precision": 1.0, "tp": 0, "fp": 0, "fn": p, "tn": n}]
    tp = fp = 0
    for cell in small:
        tp += int(cell["positive"])
        fp += int(cell["n"] - cell["positive"])
        threshold = low + int(cell["bucket"]) * (high - low) / bins if high != low else low
        points.append({
            "threshold": float(threshold), "fpr": fp / n, "tpr": tp / p,
            "recall": tp / p, "precision": tp / (tp + fp),
            "tp": tp, "fp": fp, "fn": p - tp, "tn": n - fp,
        })
    return points, {"minimum_score": low, "maximum_score": high, "positive_count": p,
                    "negative_count": n, "rows": int(statistics["rows"]), "bins": bins}


def select_threshold(validation_scores: DataFrame, bins: int = 150, beta: float = 0.5) -> dict:
    """Choose high-precision operating point on October ONLY, not the final test."""
    points, info = binned_curve(validation_scores, bins)
    candidates = []
    for point in points[1:]:
        prec, recall = point["precision"], point["recall"]
        fbeta = (1 + beta**2) * prec * recall / (beta**2 * prec + recall) if (beta**2 * prec + recall) else 0.0
        candidates.append((fbeta, point["threshold"], prec, recall))
    score, threshold, precision, recall = max(candidates, key=lambda x: (x[0], x[1]))
    return {"threshold": float(threshold), "validation_f0_5": round(float(score), 6),
            "validation_precision": round(float(precision), 6), "validation_recall": round(float(recall), 6),
            "validation_rows": info["rows"], "threshold_source": "October only; F0.5 on binned scores"}


def apply_threshold(scored: DataFrame, threshold: float) -> DataFrame:
    return scored.withColumn("decision", (F.col("score") >= F.lit(threshold)).cast("double"))


def full_metrics(scored: DataFrame, threshold: float) -> dict:
    pred = apply_threshold(scored, threshold)
    cells = {(int(row["label"]), int(row["decision"])): int(row["count"])
             for row in pred.groupBy("label", "decision").count().collect()}
    tp, tn = cells.get((1, 1), 0), cells.get((0, 0), 0)
    fp, fn = cells.get((0, 1), 0), cells.get((1, 0), 0)
    if not (tp + fn) or not (tn + fp):
        raise ValueError("One class absent from holdout set; cannot compare ROC/PR")
    safe = lambda a, b: a / b if b else 0.0
    precision = safe(tp, tp + fp)
    recall = safe(tp, tp + fn)
    specificity = safe(tn, tn + fp)
    auc_roc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="score", metricName="areaUnderROC").evaluate(scored)
    auc_pr = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="score", metricName="areaUnderPR").evaluate(scored)
    return {
        "rows": tp + tn + fp + fn, "base_rate": safe(tp + fn, tp + tn + fp + fn),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "accuracy": safe(tp + tn, tp + tn + fp + fn),
        "positive_precision": precision, "positive_recall": recall,
        "positive_f1": safe(2 * precision * recall, precision + recall),
        "specificity": specificity, "balanced_accuracy": (recall + specificity) / 2,
        "auc_roc": float(auc_roc), "auc_pr": float(auc_pr), "threshold": float(threshold),
    }


def plot_model_evidence(models: list[dict], curves: list[dict], path: Path) -> None:
    """Actual aggregate metrics only; no plot is generated until Task2 finishes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if len(models) != 4 or len(curves) != 4:
        raise ValueError("EP2 requires four actual model metrics and four held-out score curves")
    fig = plt.figure(figsize=(19, 11), constrained_layout=True)
    grid = fig.add_gridspec(3, 4, height_ratios=(0.75, 1.2, 1.2))
    summary_ax = fig.add_subplot(grid[0, :])
    summary_ax.axis("off")
    # One composite figure contains the answer sheet's required SINGLE loop of
    # all four models' best settings/times, confusion matrices, ROC and PR.
    summary = [[m["name"], str(m["best_params"]),
                f"{m['cv_seconds']:.1f}", f"{m['final_fit_seconds']:.1f}"]
               for m in models]
    table = summary_ax.table(cellText=summary, colLabels=["Model", "CV-winning parameters", "CV s", "Full fit s"],
                             colWidths=[0.20, 0.59, 0.10, 0.11], loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.9)
    confusion_axes = [fig.add_subplot(grid[1, j]) for j in range(4)]
    roc_ax = fig.add_subplot(grid[2, 0:2])
    pr_ax = fig.add_subplot(grid[2, 2:4])
    for ax, result in zip(confusion_axes, models):
        c = result["metrics"]["confusion"]
        counts = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        ax.imshow(np.log1p(counts), cmap="Blues")
        ax.set(title=result["name"], xlabel="Predicted 0 / 1", ylabel="Actual 0 / 1")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        for (i, j), value in np.ndenumerate(counts):
            ax.text(j, i, f"{value:,}", ha="center", va="center", color="#b91c1c", fontsize=9)
    for item in curves:
        roc_ax.plot([p["fpr"] for p in item["points"]], [p["tpr"] for p in item["points"]], label=item["model"])
        pr_ax.plot([p["recall"] for p in item["points"]], [p["precision"] for p in item["points"]], label=item["model"])
    roc_ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.7)
    roc_ax.set(title="Binned ROC (plot; AUC in JSON from evaluator)", xlabel="False positive rate", ylabel="True positive rate")
    pr_ax.set(title="Binned PR (plot; AUC in JSON from evaluator)", xlabel="Recall", ylabel="Precision")
    roc_ax.legend(fontsize=8); pr_ax.legend(fontsize=8)
    fig.suptitle("EP2  |  Four CV-selected Spark models: best settings, fit time, confusion, ROC and PR\n"
                 "Same untouched Nov–Dec test cohort; ROC/PR plot points binned", fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=155)
    plt.close(fig)
