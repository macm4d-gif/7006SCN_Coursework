"""Tableau-safe small aggregate extracts, link check and four-panel contact sheet.

A Tableau workbook cannot be truthfully generated/published from Python here.
Build and publish the four dashboards in Tableau Public yourself using these CSVs.
"""
from __future__ import annotations

import csv
from pathlib import Path

import requests
from pyspark.sql import SparkSession, functions as F

from .data import load_processed
from .settings import now_utc, project_path, read_json, require_verified_allocation, write_json

DASHBOARDS = [
    "Data quality & pipeline monitoring",
    "Model performance & feature importance",
    "Business insights",
    "Scalability & cost analysis",
]


def write_csv(path: Path, rows: list[dict], headers: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if headers is None:
        if not rows:
            raise ValueError(f"No records for {path}")
        headers = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as out:
        writer = csv.DictWriter(out, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def verify_public_link(url: str) -> dict:
    if not url:
        return {"valid": False, "reason": "Publish an actual 4-dashboard Tableau Public workbook; enter its URL"}
    if not url.startswith("https://public.tableau.com/"):
        return {"valid": False, "reason": "Use a direct https://public.tableau.com/ workbook link"}
    try:
        response = requests.get(url, timeout=20, allow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
        valid = response.status_code == 200 and "not-found" not in response.url.lower()
        return {"valid": valid, "status_code": response.status_code,
                "final_url": response.url, "reason": "HTTP check cannot prove four dashboards; inspect manually" if valid else "Broken/redirected link"}
    except requests.RequestException as exc:
        return {"valid": False, "reason": f"Network check failed: {type(exc).__name__}"}


def create_contact_sheet(results_dir: Path) -> str | None:
    """Combine YOUR real Tableau screenshots. No drawn/mock dashboard substitutes."""
    from PIL import Image, ImageOps, ImageDraw

    paths = [results_dir / f"tableau_dashboard_{i}.png" for i in range(1, 5)]
    if any(not f.is_file() for f in paths):
        return None
    canvas = Image.new("RGB", (2560, 1660), "#eff3f8")
    draw = ImageDraw.Draw(canvas)
    for index, (path, name) in enumerate(zip(paths, DASHBOARDS)):
        image = Image.open(path).convert("RGB")
        image.thumbnail((1220, 770))
        tile = ImageOps.pad(image, (1220, 770), color="white", centering=(0.5, 0.5))
        x, y = 25 + (index % 2) * 1265, 30 + (index // 2) * 820
        canvas.paste(tile, (x, y + 28))
        draw.text((x + 4, y + 5), f"{index + 1}. {name}", fill="#18283e")
    dest = results_dir / "task4_dashboard_contact_sheet.png"
    canvas.save(dest)
    return str(dest)


def run_task4(spark: SparkSession, cfg: dict) -> dict:
    require_verified_allocation(cfg)
    results = project_path(cfg, "results_dir")
    task1, task2, task3 = [read_json(results / f"task{i}.json") for i in (1, 2, 3)]
    if any(r.get("status") != "observed" for r in (task1, task2, task3)) or len(task2.get("models", [])) != 4:
        raise ValueError("Tasks 1-3 need complete measured results for all four models; do not export placeholders")
    data = load_processed(spark, cfg)
    quality = [
        {"section": "overall", "metric": "raw_rows", "key": "all", "value": task1["raw_row_count"]},
        {"section": "overall", "metric": "clean_rows", "key": "all", "value": task1["clean_row_count"]},
        {"section": "overall", "metric": "source_bytes", "key": "all", "value": task1["file_size_bytes"]},
        {"section": "overall", "metric": "raw_partitions", "key": "all", "value": task1["partition_count_before"]},
        {"section": "overall", "metric": "write_partitions", "key": "all", "value": task1["partition_count_after_repartition"]},
    ]
    quality += [{"section": "quality_nonexclusive", "metric": key, "key": "all", "value": val}
                for key, val in task1["quality_counts_overlapping"].items()]
    quality += [{"section": "month", "metric": "clean_rows", "key": str(month), "value": int(count)}
                for month, count in sorted({x["year_month"]: sum(int(y["count"]) for y in task1["counts_by_month_and_class"]
                                                if y["year_month"] == x["year_month"])
                                            for x in task1["counts_by_month_and_class"]}.items())]
    write_csv(results / "tbl_d1_quality.csv", quality)
    performance = []
    confusion = []
    for row in task2["models"]:
        performance.append({"model": row["name"], "family": row["family"], **{
            metric: row["metrics"][metric] for metric in
            ("auc_roc", "auc_pr", "accuracy", "positive_precision", "positive_recall", "positive_f1", "specificity")},
            "cv_seconds": row["cv_seconds"], "refit_seconds": row["final_fit_seconds"],
            "folds": row["cv_folds"], "selected_threshold": row["threshold_tuning"]["threshold"]})
        confusion += [{"model": row["name"], "cell": cell, "count": n}
                      for cell, n in row["metrics"]["confusion"].items()]
    write_csv(results / "tbl_d2_performance.csv", performance)
    write_csv(results / "tbl_d2_confusion.csv", confusion)
    curves = read_json(results / "task2_curves.json")
    curve_rows = [{"model": series["model"], "fpr": p["fpr"], "tpr": p["tpr"],
                   "recall": p["recall"], "precision": p["precision"], "threshold": p["threshold"]}
                  for series in curves["curves"] for p in series["points"]]
    write_csv(results / "tbl_d2_curves.csv", curve_rows)
    global_rows = [{"model": item["name"], **importance}
                   for item in task2["models"]
                   for importance in item.get("tree_global_split_importance", [])]
    if not global_rows:
        raise ValueError("Task2 must export fitted RF/GBT GLOBAL split importance; rerun the updated Task2")
    write_csv(results / "tbl_d2_global_tree_importance.csv", global_rows)
    write_csv(results / "tbl_d2_lime.csv", task3["explainability"]["feature_weights"])
    minimum = int(cfg["model"]["minimum_borough_size"])
    # Spark aggregates the real 2019 dataset. No trip-level records are published.
    groups = (data.groupBy("year_month", "pickup_borough", "pickup_hour")
              .agg(F.count(F.lit(1)).alias("n"), F.avg("label").alias("high_tip_rate"),
                   F.avg("fare_amount").alias("mean_fare_usd"))
              .where(F.col("n") >= minimum).orderBy("year_month", "pickup_borough", "pickup_hour").collect())
    business = [{"month": g["year_month"], "borough": g["pickup_borough"], "pickup_hour": g["pickup_hour"],
                 "n": g["n"], "high_tip_rate": g["high_tip_rate"], "mean_fare_usd": g["mean_fare_usd"]} for g in groups]
    write_csv(results / "tbl_d3_business.csv", business)
    distance = (data.withColumn("distance_band", F.when(F.col("trip_distance") < 2, "0-2 miles")
                .when(F.col("trip_distance") < 5, "2-5 miles")
                .when(F.col("trip_distance") <= 200, "5-200 miles").otherwise("unknown/outlier"))
                .groupBy("distance_band").agg(F.count(F.lit(1)).alias("n"), F.avg("label").alias("rate"))
                .where(F.col("n") >= minimum).collect())
    write_csv(results / "tbl_d3_distance_DESCRIPTIVE_ONLY.csv", [x.asDict() for x in distance])
    write_csv(results / "tbl_d3_borough_audit.csv", task3["fairness"]["groups"])
    # Cost calculations ONLY if the user supplies an auditable price and source.
    price = cfg["tableau"].get("hourly_cluster_cost_usd")
    price_source = cfg["tableau"].get("cluster_cost_source", "")
    if price is not None and (float(price) <= 0 or not price_source.startswith("https://")):
        raise ValueError("Provide a positive actual cluster price AND an HTTPS price/source URL, or set both blank")
    scale = []
    for row in task2["models"]:
        total = row["cv_seconds"] + row["final_fit_seconds"]
        scale.append({"experiment": "training", "model": row["name"], "config": "CV + refit",
            "wall_seconds": round(total, 3), "usd_estimate": round(total / 3600 * float(price), 4) if price is not None else "",
            "cost_source": price_source if price is not None else "unpriced"})
    for trial in task3["optimisations"]:
        if trial["change"] == "shuffle_partition_count":
            scale.append({"experiment": "shuffle", "model": "Spark groupBy", "config": str(trial["partitions"]),
                "wall_seconds": round(trial["median_seconds"], 3), "usd_estimate": "", "cost_source": "benchmark only"})
        else:
            scale.extend([
                {"experiment": "cache", "model": "Spark groupBy", "config": "uncached repeated median",
                 "wall_seconds": trial["median_before_seconds"], "usd_estimate": "", "cost_source": "benchmark only"},
                {"experiment": "cache", "model": "Spark groupBy", "config": "cached repeated median excludes fill",
                 "wall_seconds": trial["median_after_seconds"], "usd_estimate": "", "cost_source": "benchmark only"},
                {"experiment": "cache", "model": "Spark groupBy", "config": "cache fill",
                 "wall_seconds": trial["cache_fill_seconds"], "usd_estimate": "", "cost_source": "benchmark only"},
            ])
    for point in task3.get("scalability_query_benchmarks", []):
        scale.append({"experiment": "groupby_fraction", "model": "Spark groupBy",
            "config": f"sample={point['sample_fraction']}",
            "sample_fraction": point["sample_fraction"], "rows": point["rows"],
            "wall_seconds": point["group_by_seconds"], "usd_estimate": "", "cost_source": "benchmark only"})
    if price is not None:
        for measurement in scale:
            measurement["usd_estimate"] = round(float(measurement["wall_seconds"]) / 3600 * float(price), 4)
            measurement["cost_source"] = price_source + " (cluster rate only; no storage/egress)"
    write_csv(results / "tbl_d4_scalability.csv", scale,
              headers=["experiment", "model", "config", "sample_fraction", "rows", "wall_seconds", "usd_estimate", "cost_source"])
    link = verify_public_link(cfg["tableau"].get("public_workbook_url", ""))
    contact = create_contact_sheet(results)
    declared = bool(cfg["tableau"].get("published_four_dashboards_verified", False))
    published = bool(link["valid"] and contact and declared)
    out = {
        "status": "observed" if published else "pending_publication_or_screenshots",
        "generated_utc": now_utc(), "tableau_public_link": cfg["tableau"].get("public_workbook_url", ""),
        "link_check": link, "dashboards": DASHBOARDS if published else [],
        "dashboard_titles_planned": DASHBOARDS,
        "contact_sheet": contact,
        "screenshot_requirement": "Export a real PNG per dashboard named results/tableau_dashboard_1.png ... _4.png",
        "exports": [str(p.name) for p in sorted(results.glob("tbl_d*.csv"))],
        "descriptive_not_predictive": "Trip-distance extract is retrospective; distance is intentionally NOT a pickup-time model feature",
        "price_per_hour_usd": price,
        "limitation_note": "Only electronic-payment reported tips; grouped insights are observational; no causal driver incentives from these charts alone.",
    }
    write_json(results / "task4.json", out)
    return out
