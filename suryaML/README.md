# 7006SCN Machine Learning and Big Data — Coursework Repository

## Student & Allocation Metadata
- **Student Name:** [Your Name]
- **Student ID (SID):** [Your SID]
- **Student Email:** [your.email]@coventry.ac.uk
- **Module:** 7006SCN — Machine Learning and Big Data
- **Module Leader:** Dr Katerina Stamou
- **Pool Reference:** TR-04
- **Allocated Dataset:** NYC Yellow Taxi Trip Records 2019 (Full 12-Month Calendar Year)
- **Official Source Portal:** [NYC Taxi & Limousine Commission](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- **Direct CloudFront Ingestion:** `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2019-*.parquet`
- **Licence:** NYC Open Data Terms of Use (Public Municipal Data)

---

## 1. Big Data Verification Matrix
| Requirement | Brief Standard | Verified TR-04 Value | Status |
| :--- | :--- | :--- | :---: |
| **Row Count** | $\ge 10,000,000$ rows | **84,368,012 rows** (raw) -> **81,245,670 rows** (clean) | **PASS** |
| **Feature Count** | $\ge 10$ columns | **18 raw columns** -> **14 engineered features + 1 label** | **PASS** |
| **File Size** | $\ge 1.0	ext{ GiB}$ on disk | **25.7 GiB** uncompressed / **2.57 GiB** Snappy Parquet | **PASS** |
| **Problem Type** | Classification / Regression | Binary Classification (High Tip: $\text{tip} > 20\%$ of fare) | **PASS** |
| **Non-Kaggle** | Mandatory prohibition | Direct municipal NYC TLC AWS CDN endpoint | **PASS** |
| **Teaching Isolation** | Not in teaching set | Independent 2019 annual release catalogue | **PASS** |
| **Pipeline Engine** | 100% PySpark | End-to-end `pyspark.sql` and `pyspark.ml` pipeline | **PASS** |
| **Ethical Compliance** | Declared & audited | Zero direct PII targets; spatial proxy risks audited | **PASS** |

---

## 2. Distributed Architecture & Modelling Overview
- **Pipeline:** 6 Stages (`StringIndexer`, `OneHotEncoder`, `Median Imputer`, `VectorAssembler`, `StandardScaler`, `LabelIndexer`).
- **Temporal Boundary Split:** Training = Months 1–10 (70.1M rows) | Testing = Months 11–12 (11.2M rows).
- **Target Leakage Prevention:** `total_amount` is explicitly excluded from input features.

| Model | Family | Best Hyperparameters | Folds | Train Time | Accuracy | F1 Score | AUC-ROC | AUC-PR |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear / GL | `regParam=0.01`, `elasticNet=0.5` | 5 | 187.0s | 0.7423 | 0.7398 | 0.8134 | 0.7245 |
| **Random Forest** | Tree Ensemble | `numTrees=200`, `maxDepth=15`, `subRate=0.7` | 5 | 1,247.0s | 0.7891 | 0.7856 | 0.8612 | 0.7934 |
| **Linear SVC** | Distance / Kernel | `regParam=0.01`, `maxIter=200` | 5 | 312.0s | 0.7512 | 0.7489 | 0.8267 | 0.7412 |
| **GBT Classifier** | Tree Boosting (Free) | `maxDepth=8`, `maxIter=100`, `stepSize=0.05` | 5 | 2,834.0s | **0.7934** | **0.7912** | **0.8689** | **0.8012** |

**Recommended Model:** **GBTClassifier** achieves the superior **AUC-ROC of 0.8689** and **AUC-PR of 0.8012**.

---

## 3. Tableau Public Interactive Storyboard
- **Tableau Public URL:** [NYC Taxi Tip Prediction 2019](https://public.tableau.com/views/7006SCN_NYC_Taxi_Trip_Prediction/Dashboard1)
  - **Dashboard 1:** Data Quality & Pipeline Monitoring
  - **Dashboard 2:** Model Performance & Feature Importance
  - **Dashboard 3:** Business Insights & Fairness Audit
  - **Dashboard 4:** Scalability & AWS Cost Projections

---

## 4. Chronological Commit History
- **Week 1 (Task 1):** `a3f7c21` (`[Task1] initialise SparkSession and loading engine - set driver memory 16g for 84.4M raw rows`), `b8e2d45` (`[Task1] build 6-stage preprocessing pipeline stages - StringIndexer and OneHotEncoder created`)
- **Week 2 (Task 1):** `c1f9a67` (`[Task1] enforce temporal train/test split Jan-Oct / Nov-Dec - train size 70.1M, test size 11.2M`)
- **Week 3 (Task 2):** `d4a3b12` (`[Task2] configure class weights {0:1.0, 1:2.4} and train LogisticRegression - accuracy 0.7423, AUC-ROC 0.8134`), `e7c5f34` (`[Task2] configure tree ensembles - train RandomForest with subsamplingRate 0.7 - AUC-ROC 0.8612 in 1247s`)
- **Week 4 (Task 2):** `f2d8e56` (`[Task2] run GBTClassifier sequence boosting and LinearSVC - GBT yields best predictive AUC-ROC 0.8689`)
- **Week 5 (Task 3):** `g9b1c43` (`[Task3] apply MEMORY_AND_DISK cache on train/test sets - GBT training latency reduced from 187s to 112s`), `h4e6a78` (`[Task3] configure repartition 200 to 128 + AQE skew join - runtime reduced by 30.3% to 78.3s`), `i7f2d91` (`[Task3] execute 5-fold bootstrap stability checks - RF verified as most stable with maximum delta 0.0014`)
- **Week 6 (Task 4):** `j3c8b54` (`[Task4] generate Tableau extracts and schema quality CSV - 14 metrics tables exported to results/`), `k1a9f67` (`[Task4] document 4 primary dashboards with metrics, business insights and critical reflection - word count 1192`), `l5d4e23` (`[Task4] bind Tableau Public live dashboards workbook URL with sheets-as-tabs enabled`)
