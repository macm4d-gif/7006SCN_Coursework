# How this project mirrors `saiashishd/collegework` — and why it cannot be a clone



| Reference stage | New TR-04 counterpart | Important change required by *this* brief |
|---|---|---|
| One main `.py` | `NYC_Taxi_TR04.py --plan` shows the complete 12-stage numbered Spark workflow; the executable CLI calls tested code in `coursework/`. `scripts/run_project.py` is a compatible alias. | Four numbered Task notebooks are still compulsory. Keep heavy logic modular/testable without copying the old notebook's history or hard-coded resources. |
| Spark configuration and CSV validation | `Task1.ipynb`, official 12-month Parquet downloader, schema/type normalisation, real rows/bytes/partitions | NYC TLC Parquet is the confirmed allocation; road-collision CSV is not. Neither the old executor settings nor row counts are transferable. |
| Parquet, domain features and VectorAssembler | `data.py` and `models.py`: borough/time/temporal features, missing-value imputation, index/one-hot/assembler/scaler stages | Cash tips aren't reported; train on card-only cohort, and exclude tip/fare/total/drop-off/distance/final RatecodeID from pickup-time predictors. Fit prep INSIDE CV to avoid leakage. |
| Random 80/20 split | Jan–Sep fitting/CV, October threshold, Nov–Dec untouched test | Future generalisation matters; random split would flatter temporal performance. |
| Four regressors; only forest tuned; RMSE | LogisticRegression, RandomForestClassifier, MultilayerPerceptronClassifier and GBTClassifier, **each** tuned with CrossValidator; PR/ROC AUC and confusion | Four classifiers explicitly cover linear, tree-ensemble, **neural** (third required family) and free-choice boosting. Spark MLP does not expose a weightCol: describe its unweighted objective. Binary classification does not use the old regression RMSE or accident labels. |
| Cache/scalability statements | Task3 repeated identical-action timing, cache-fill cost, shuffle/partition benchmarks and 4-model stability deltas | Source README claims aren't substitutes for your own Spark UI and cluster timings. |
| Four `.twbx` dashboard files | Task4 Spark-generated aggregate CSVs, dashboard build specification and **one four-dashboard Tableau Public workbook** to create/publish with your own account | This brief demands Tableau **Public**, not the old Tableau Online workbooks, and requires four specific dashboard themes and a real contact sheet. No template `.twbx` is passed off as a published dashboard. |
| README summary | README + task JSON provenance + Word-draft assembler that requires **your authored interpretations** | Results, private-org commit history, live notebook links and AI-use declaration must be genuine. |

**Structural similarity:** high (numbered Spark setup → twelve-file ingestion/validation → domain features → Parquet → bounded caching → fold-safe assembler → time split → four models → four CrossValidators → actual Spark model serialization → source-derived EDA → Tableau). The four notebooks now contain **10 clearly numbered steps in each of four Task notebooks** and actual plotting/summary cells, analogous in reading order to the user's Colab notebook and `.py` export. **Direct content reuse:** none. Your new repo must be private inside `7006SCN2627SEPNOV`; do not submit the personal public collision repo, repurpose its dataset, copy its scores, or use its Tableau Online links.

## Specific checks on the pasted Colab-generated `.py` file

- Its `SparkSession.builder` settings (4 executors × 6 GB, 400 shuffle partitions) are **requested values**, not proof of actual cluster resources or an optimisation. The taxi project reads *your* approved cluster config, leaves the executor count to the cluster by default, logs the live application config and benchmarks settings in Task 3.
- It reads a UK road-collision CSV with `collision_severity` as a numerical label and uses four **regressors with RMSE**. TR-04 instead reads the twelve allocated NYC TLC Parquets and fits four **classifiers** using ROC/PR AUC, confusion, precision/recall and real fit costs.
- It only cross-validates Random Forest; the taxi project runs a real `CrossValidator` **for each of four model families**, with fold-fitted feature preprocessing and a held-out future period.
- The heading “model serialization” in that script does **not** serialise a model: it trains an unrelated scikit-learn RF on a 50,000-row driver sample. Taxi Task 2 saves actual trained Spark `PipelineModel`s and its step 🔟 reloads the chosen model.
- Its two displayed row totals (**706,953** versus **248,497**) disagree, and the later printed schema includes `collision_index` absent from the initial `select`. They are stale/inconsistent cell outputs, not evidence for any TR-04 count. A repeated weather-condition cell and unconditional `.fillna(0)` also obscure the processing record. The new notebooks start with **empty** outputs and compute row counts/plots from the student's actual Spark run; they never import these old figures.

The old `.py` and its notebook are **presentation references only**. Do not copy their data, scores, notebooks, workbooks, code provenance or test output into an assessed NYC Taxi submission.
