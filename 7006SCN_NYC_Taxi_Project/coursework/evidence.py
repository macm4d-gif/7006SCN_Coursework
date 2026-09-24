"""Figures assembled only from observed Spark results; no example outcomes."""
from __future__ import annotations

from pathlib import Path


def task1_evidence_pack(result: dict, path: Path) -> None:
    """One ordered EP1 image, rerenderable in a SINGLE Task1 notebook cell.

    The small preprocessing-only demonstrator is FIT on Jan--Sep rows. It is
    explicitly distinct from the four later Task2 models, which fit new
    preprocessors within CV. Neither test data nor invented stage lists enter.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    identity = result["student"]
    cfg = result["spark_configuration"]
    lines = [
        "EP1  |  ACTUAL SPARK INGESTION AND PIPELINE EVIDENCE",
        f"Dataset: {result['dataset_name']}  |  Pool: {result['pool_reference']}",
        f"Student: {identity['name']}  |  Email: {identity['email']}",
        "",
        "1. SparkSession initialisation (actual running application)",
        f"   Spark {cfg['spark_version']}  |  master {cfg['spark.master']}  |  app ID {cfg['application_id']}",
        "",
        "2. Original January df.printSchema() (not a hand-authored schema)",
        *["   " + line for line in result["raw_january_print_schema"].strip().splitlines()],
        "",
        "3. Raw row and column counts (all 12 monthly Parquet sources)",
        f"   {result['raw_row_count']:,} rows  |  {result['raw_column_count']} original January columns",
        "",
        "4. Actual file-size verification of these same source files",
        f"   {result['source_file_count']} files  |  {result['file_size_bytes']:,} bytes  |  {result['file_size_gib']} GiB",
        "",
        "5. Partition counts: raw ingestion -> measured post-repartition plan",
        f"   {result['partition_count_before']} -> {result['partition_count_after_repartition']}",
        "",
        "6. FITTED preprocessing-only PipelineModel stages (ordered)",
        f"   Demonstrator: {result['preprocessing_fit_sample_rows']} Jan--Sep training rows, limit {result['preprocessing_fit_limit']}.",
        *[f"   {i:02d}  {name}" for i, name in enumerate(result["preprocessing_stages"], 1)],
        "   Task2 separately fits each model's pipeline within CV and on full Jan--Sep.",
    ]
    fig, ax = plt.subplots(figsize=(14, max(11.5, len(lines) * 0.245)))
    ax.axis("off")
    ax.text(0.02, 0.985, "\n".join(lines), family="monospace", fontsize=9.3,
            va="top", ha="left", transform=ax.transAxes, linespacing=1.15)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=145, bbox_inches="tight", facecolor="white")
    plt.close(fig)
