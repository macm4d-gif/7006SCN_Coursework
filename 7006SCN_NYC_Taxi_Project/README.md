# 7006SCN — NYC Yellow Taxi Tip Prediction

## Overview

This project develops an end-to-end **PySpark machine-learning pipeline** to predict whether a credit-card NYC Yellow Taxi trip will result in a **tip greater than 20% of the fare**.

Using the allocated **TR-04 — 2019 NYC Yellow Taxi** dataset, the project covers large-scale data processing, feature engineering, exploratory analysis, machine learning, model evaluation, performance benchmarking, explainability, stability analysis and Tableau visualisation.

All final metrics, screenshots, dashboards and experimental results must be generated from genuine project executions. No grade is guaranteed.

## Research Question

> Can pickup-time and trip-context information predict whether a credit-card taxi trip will receive a tip greater than 20% of the fare?

### Features

* Pickup borough
* Pickup hour
* Pickup day
* Pickup month
* Vendor
* Passenger count

The model excludes realised-trip variables such as `tip_amount`, `fare_amount`, `total_amount`, `trip_distance`, drop-off information and `payment_type` to reduce target leakage.

## Temporal Evaluation

| Period       | Purpose                     |
| ------------ | --------------------------- |
| Jan–Sep 2019 | Training & cross-validation |
| October 2019 | Threshold selection         |
| Nov–Dec 2019 | Final evaluation            |

The November–December data remains untouched until final evaluation.

## Machine Learning Models

Four Spark ML classifiers are implemented:

1. **Logistic Regression**
2. **Random Forest**
3. **Multilayer Perceptron**
4. **Gradient-Boosted Trees**

Models use Spark ML pipelines, cross-validation and hyperparameter tuning.

## Evaluation

Models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Balanced Accuracy
* ROC-AUC
* PR-AUC
* Confusion Matrix

ROC and Precision–Recall curves and tree-model feature importance are also analysed.

## Project Tasks

### Task 1 — Data Engineering & EDA

* Ingest and validate all 12 monthly files.
* Perform data-quality checks.
* Create the target and features.
* Generate partitioned Parquet data.
* Conduct exploratory analysis.

### Task 2 — Machine Learning

* Train four classification models.
* Perform 3-fold cross-validation.
* Tune hyperparameters.
* Refit selected models on January–September data.
* Evaluate on November–December data.

### Task 3 — Performance & Explainability

* Benchmark Spark performance and caching.
* Perform model stability analysis.
* Apply LIME explainability.
* Analyse geographic prediction patterns.
* Capture genuine Spark UI evidence.

### Task 4 — Tableau Analytics

* Generate aggregated analytical datasets.
* Create four Tableau dashboards.
* Publish dashboards to Tableau Public.
* Capture genuine dashboard screenshots.

## Project Structure

```text
NYC_Taxi_TR04/
├── NYC_Taxi_TR04.py
├── notebooks/
│   ├── Task1.ipynb
│   ├── Task2.ipynb
│   ├── Task3.ipynb
│   └── Task4.ipynb
├── coursework/
├── scripts/
├── config/
├── report/
├── results/
├── requirements.txt
└── README.md
```

## Technologies

* Python
* PySpark / Apache Spark
* Spark MLlib
* Spark SQL
* LIME
* Jupyter Notebook
* Tableau Public
* Git / GitHub
* Pytest

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp config/config.university.example.json config/config.json

python scripts/download_tlc.py --config config/config.json
pytest -q
python NYC_Taxi_TR04.py --plan
jupyter lab
```

Run the notebooks in order:

```text
Task 1 → Task 2 → Task 3 → Task 4
```

The full dataset should be processed on a suitable Spark cluster or sufficiently resourced environment rather than an underpowered local machine.

## Reproducibility & Integrity

All reported results must come from actual executions. The project must not fabricate:

* Metrics
* Screenshots
* Spark UI evidence
* Tableau results
* Git commits
* Runtime measurements
* Business impacts

AI assistance must also be disclosed accurately in accordance with the university requirements.

## Data & Ethics

The project acknowledges potential data-quality, payment-selection and geographic/socioeconomic proxy effects. Geographic comparisons are treated as descriptive analysis rather than causal or protected-group disparity conclusions.

---

**Project:** 7006SCN — Machine Learning and Big Data
**Dataset:** NYC Yellow Taxi — 2019, TR-04
**Framework:** Apache Spark / PySpark
**Approach:** End-to-end distributed machine learning
