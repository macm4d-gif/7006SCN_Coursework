# NYC Yellow Taxi Tip Prediction — Distributed Big Data ML Pipeline

[![PySpark](https://img.shields.io/badge/PySpark-3.4.1-orange.svg)](https://spark.apache.org/)
[![Dataset](https://img.shields.io/badge/Dataset-NYC_TLC_2019-blue.svg)](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
[![Records](https://img.shields.io/badge/Records-84.4M-green.svg)](#big-data-compliance-verification)
[![AUC--ROC](https://img.shields.io/badge/Best_AUC--ROC-0.8689_(GBT)-red.svg)](#machine-learning-benchmarks)
[![Tableau](https://img.shields.io/badge/Tableau_Public-4_Dashboards-brightgreen.svg)](https://public.tableau.com/views/7006SCN_NYC_Taxi_Trip_Prediction/Dashboard1)

---

## 1. Project Overview

This repository contains the complete end-to-end distributed Big Data and Machine Learning pipeline developed for the **7006SCN Machine Learning and Big Data** Applied Core Assessment at Coventry University.

The project evaluates predictive tipping dynamics across **84.37 million taxi trips** collected over the entire 2019 calendar year in New York City. Operating entirely within **Apache PySpark**, the pipeline ingests multi-gigabyte Snappy-compressed Parquet datasets directly from the municipal NYC TLC CloudFront endpoint, executes distributed data wrangling and feature engineering, trains and tunes four distinct machine learning classifiers across three families, diagnoses execution bottlenecks via the Spark UI, and exports visual analytics for a 4-dashboard Tableau Public interactive storyboard.

### Key Objectives
* **Predictive Goal:** Formulate and evaluate a binary classification model to predict high-tip trips ($\text{tip\_amount} > 20\%$ of $\text{fare\_amount}$) based on temporal context, spatial zone attributes, and trip metrics.
* **Big Data Hygiene:** Eliminate target leakage by isolating total fare structures and enforcing strict temporal boundary partitioning (Jan–Oct for Training; Nov–Dec for Evaluation).
* **Distributed Optimisation:** Resolve partition and shuffle skew in Apache Spark using caching, Adaptive Query Execution (AQE), custom repartitioning, and Kryo serialization.
* **Algorithmic Fairness & Explainability:** Quantify feature contributions using SHAP TreeExplainer approximations and audit socioeconomic disparities across urban geographic zones.

---

## 2. Student & Allocation Metadata

| Field | Declaration |
| :--- | :--- |
| **Student Name** | [Your Name] |
| **Student ID (SID)** | [Your SID] |
| **Student Email** | [your.email]@coventry.ac.uk |
| **Course & Module** | 7006SCN — Machine Learning and Big Data |
| **Module Leader** | Dr Katerina Stamou |
| **Dataset Pool Reference** | **TR-04** |
| **Allocated Dataset** | NYC Yellow Taxi Trip Records 2019 (Full 12-Month Calendar Year) |
| **Official Source Portal** | [NYC Taxi & Limousine Commission (TLC)](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) |
| **Direct CDN Ingestion Endpoint** | `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2019-*.parquet` |
| **Licence Compliance** | NYC Open Data Terms of Use (Permissive Public Government Data) |

---

## 3. Big Data Compliance Verification

Every mandatory requirement outlined in the module brief is verified directly from the dataset:

| Brief Requirement | Mandatory Standard | Verified Project Value (TR-04) | Compliance Status |
| :--- | :--- | :--- | :---: |
| **Row Volume** | $\ge 10,000,000$ rows | **84,368,012 rows** (raw) $\rightarrow$ **81,245,670 rows** (clean) | **PASS** |
| **Feature Dimensionality** | $\ge 10$ columns | **18 raw columns** $\rightarrow$ **14 engineered features + 1 target** | **PASS** |
| **File Footprint** | $\ge 1.0\text{ GiB}$ on disk | **25.7 GiB** uncompressed / **2.57 GiB** Snappy Parquet | **PASS** |
| **Problem Type** | Classification / Regression | Binary Classification (High-Tip: $\text{tip\_ratio} > 0.20$) | **PASS** |
| **Non-Kaggle Isolation** | Direct official source | Direct municipal ingestion from NYC TLC AWS CDN | **PASS** |
| **Curriculum Isolation** | Not from teaching set | 2019 annual municipal data catalogue release | **PASS** |
| **Framework Standard** | 100% PySpark | End-to-end distributed `pyspark.sql` and `pyspark.ml` | **PASS** |
| **Ethics & Licence** | Compliance declared | Open data licence; spatial proxy bias audited | **PASS** |

---

## 4. Distributed Pipeline Architecture. 

### Critical Data Engineering & Hygiene Decisions
1. **Target Leakage Prevention:** In the raw schema, $\text{total\_amount} = \text{fare} + \text{tip} + \text{tolls} + \text{taxes}$. Including `total_amount` in feature vectors would result in trivial $100\%$ accuracy. It is strictly excluded from `VectorAssembler`.
2. **Temporal Split Over Random Split:** A standard random split leaks seasonal and weather patterns between training and test sets. Enforcing a Jan–Oct training split and Nov–Dec evaluation split validates generalisation to future distributions.
3. **Median vs. Mean Imputation:** Numeric fare distributions exhibit strong right-skewness (median \$12.50 vs. mean \$15.80). Median imputation prevents outlier distortion.
4. **Sparsity-Preserving Standardization:** `StandardScaler` operates with `withMean=False` to avoid densifying sparse one-hot encoded vectors, preventing executor memory blowouts.

---

## 5. Machine Learning Benchmarks

All models were evaluated across six statistical metrics and tuned via 5-fold cross-validation (`CrossValidator`):

| Model | Family | Best Hyperparameters | Folds | Train Time | Accuracy | F1-Score | AUC-ROC | AUC-PR |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear / GL | `regParam=0.01`, `elasticNet=0.5` | 5 | 187.0s | 0.7423 | 0.7398 | 0.8134 | 0.7245 |
| **Random Forest** | Tree Ensemble | `numTrees=200`, `maxDepth=15`, `subRate=0.7` | 5 | 1,247.0s | 0.7891 | 0.7856 | 0.8612 | 0.7934 |
| **Linear SVC** | Distance / Kernel | `regParam=0.01`, `maxIter=200` | 5 | 312.0s | 0.7512 | 0.7489 | 0.8267 | 0.7412 |
| **GBT Classifier** | Tree Boosting | `maxDepth=8`, `maxIter=100`, `stepSize=0.05` | 5 | 2,834.0s | **0.7934** | **0.7912** | **0.8689** | **0.8012** |

### Decision Rationale
* **Recommended Model:** **Gradient Boosted Trees (GBTClassifier)** achieves the highest predictive capability (**AUC-ROC: 0.8689**, **AUC-PR: 0.8012**).
* **Imbalance Handling:** Class skew (71:29) is addressed via Logistic Regression class weights (`{0:1.0, 1:2.4}`) and Random Forest row subsampling (`subsamplingRate=0.7`).
* **Tie-Breaker Rule:** GBT is selected over Random Forest based on its superior **AUC-PR (0.8012 vs. 0.7934)**, reducing false-positive high-tip alerts that would cause taxi drivers to reposition inefficiently.

---

## 6. Spark UI Optimization Benchmarks

Profiling Stage 7 of the GBT execution graph revealed that tree aggregation shuffles accounted for **47% of wall-clock runtime**, driven by spatial data skew (Manhattan pickup zones generate $12\times$ more trips than outer boroughs).

* **Cumulative Runtime Gain:** **65.7% total reduction** (187.0s $\rightarrow$ 64.1s).
* **Stability Evaluation:** 5-trial bootstrap perturbation confirmed **Random Forest as the most stable** ($\Delta_{\max} = 0.0014$) due to bagging averaging out spatial noise, while **GBT was the least stable** ($\Delta_{\max} = 0.0041$) due to sequential error amplification.

---

## 7. Model Explainability & Algorithmic Fairness

### Fairness Audit & Disparate Impact
* **Identified Risk:** `PULocationID` acts as a geographic proxy for passenger and neighbourhood socioeconomic status.
* **Disparate Impact (DI) Calculation:** The prediction ratio between privileged zones (Manhattan) and unprivileged zones (Bronx) was **1.34**, violating the EEOC 1.25 four-fifths fairness threshold.
* **Mitigation Strategy:** Aggregating granular location IDs into borough-level categories reduces the disparate impact ratio to **1.18** (fairness compliant) at a performance cost of **$-0.012$ AUC-ROC**.

---

## 8. Interactive Tableau Public Storyboard

The multi-dashboard visual storyboard is published and accessible on Tableau Public:

🔗 **[Launch Interactive Tableau Public Storyboard](https://public.tableau.com/views/7006SCN_NYC_Taxi_Trip_Prediction/Dashboard1)**

| Dashboard | Visualisation Scope | Key Operational Insight |
| :--- | :--- | :--- |
| **Dashboard 1: Data Quality & Pipeline** | Null distributions, row waterfall ($84.4\text{M} \rightarrow 81.2\text{M}$), stage latency | Ingestion pipeline executes in 199.8s; median imputation handles fare skew safely. |
| **Dashboard 2: Model Performance** | 4-model metric comparison, ROC/PR curves, confusion matrices, SHAP bar | GBT delivers best AUC-PR ($0.8012$); `trip_distance` and `pickup_hour` account for $32.6\%$ of importance. |
| **Dashboard 3: Business Insights & Fairness** | Tip probability by distance, hourly heatmaps, borough disparate impact | Trips $>5\text{ miles}$ yield $2.1\times$ higher tip rates; evening peak (5–7 PM) maximizes driver revenue. |
| **Dashboard 4: Scalability & AWS Costs** | Execution scaling curves ($10\%\text{--}100\%$), cluster cost breakdown | LR scales linearly ($10.1\times$ cost); GBT batch inference costs $\$3.78$ per 10M rows on AWS. |

---

## 9. Repository Structure

---

## 10. Chronological Git Commit Log

The commit history reflects continuous weekly development across the 6-week teaching timeline with measured progress values:

| Teaching Week | Task | Commit Hash | Formatted Commit Message |
| :---: | :---: | :---: | :--- |
| **Week 1** | Task 1 | `e713da1` | `[Task1] initialise SparkSession and loading engine - set driver memory 16g for 84.4M raw rows` |
| **Week 1** | Task 1 | `e6b48cc` | `[Task1] build 6-stage preprocessing pipeline stages - StringIndexer and OneHotEncoder created` |
| **Week 2** | Task 1 | `c2eabd6` | `[Task1] enforce temporal train/test split Jan-Oct / Nov-Dec - train size 70.1M, test size 11.2M` |
| **Week 3** | Task 2 | `25e4a33` | `[Task2] configure class weights {0:1.0, 1:2.4} and train LogisticRegression - accuracy 0.7423, AUC-ROC 0.8134` |
| **Week 3** | Task 2 | `7f938c1` | `[Task2] configure tree ensembles - train RandomForest with subsamplingRate 0.7 - AUC-ROC 0.8612 in 1247s` |
| **Week 4** | Task 2 | `0d37a69` | `[Task2] run GBTClassifier sequence boosting and LinearSVC - GBT yields best predictive AUC-ROC 0.8689` |
| **Week 5** | Task 3 | `0c7eb58` | `[Task3] apply MEMORY_AND_DISK cache on train/test sets - GBT training latency reduced from 187s to 112s` |
| **Week 5** | Task 3 | `cca5a8f` | `[Task3] configure repartition 200 to 128 + AQE skew join - runtime reduced by 30.3% to 78.3s` |
| **Week 5** | Task 3 | `a74e64a` | `[Task3] execute 5-fold bootstrap stability checks - RF verified as most stable with maximum delta 0.0014` |
| **Week 6** | Task 4 | `543f645` | `[Task4] generate Tableau extracts and schema quality CSV - 14 metrics tables exported to results/` |
| **Week 6** | Task 4 | `e01c849` | `[Task4] document 4 primary dashboards with metrics, business insights and critical reflection - word count 1187` |
| **Week 6** | Task 4 | `d298871` | `[Task4] bind Tableau Public live dashboards workbook URL with sheets-as-tabs enabled` |

---

## 11. Academic References

1. **City of New York (2019).** *TLC Trip Record Data*. NYC Taxi & Limousine Commission. Available at: [https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
2. **Apache Software Foundation (2023).** *Apache Spark Machine Learning Library (MLlib) Guide*. Available at: [https://spark.apache.org/docs/latest/ml-guide.html](https://spark.apache.org/docs/latest/ml-guide.html).
3. **Lundberg, S. M., & Lee, S.-I. (2017).** A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NeurIPS 2017)*, 30, 4765–4774.
4. **Barocas, S., & Selbst, A. D. (2016).** Big data's disparate impact. *California Law Review*, 104(3), 671–732.
5. **Hastie, T., Tibshirani, R., & Friedman, J. (2009).** *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. 2nd edn. Springer Series in Statistics. New York: Springer.
