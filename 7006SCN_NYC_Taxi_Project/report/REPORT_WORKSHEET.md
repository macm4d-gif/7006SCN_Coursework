# Report worksheet — fill with YOUR evidence after running the notebooks

Do not submit this worksheet as an answered report. The uploaded file named `mlrepot.pdf` was actually a Word OOXML document; its byte-for-byte copy is `report/7006SCN_Official_Answer_Sheet_TEMPLATE.docx`. Copy that file to a new `.docx`, KEEP all its subsections/tables and complete it as your ONE official submission. Do not submit the still-blank template. For automatic layout from genuine task JSON plus YOUR OWN analysis, complete `report/author_content.example.json` as `report/author_content.json` and run `python scripts/build_report.py`; that generates a DRAFT to review/transfer to the official answer sheet, never an automatically authored analysis. The brief specifies 1200 ±10%, so stay between **1080 and 1320 words**; it also contains an inconsistent parenthetical reference to 2200 words. Ask the Module Leader if uncertain, and state the counted number at the START and end. Report front matter: student identity, **verified** pool reference, source, licence/terms, actual private module-organisation repo URL. Embed genuine figures and active private-org notebook links plus a live Tableau Public workbook link. Paste YOUR signed MS Forms checklist screenshot into the FIRST evidence-image cell, add four REAL Tableau screenshots as a contact sheet, and keep the appendix prompt log. State `Total body word count: NNNN` near the top of the answer sheet, and fill the per-task word counts. Every analytical claim needs a numbered figure/table from your own notebook; each justified method should name its taught Day/session; supply genuine local commit hashes for all 11 numbered subsections. Do not recycle unsupported claims, fictitious git hashes or someone else's AI-use statement.

## Report heading -> AI Use Declaration (required)

Write your TRUE account of which AI tools assisted, how, in which sections/code/figures, and what you actually reviewed, changed and verified. If you use this project: acknowledge **Arena.ai Agent Mode**'s contributions to `coursework/`, notebooks, instructions and any report wording you adapt. If you did not test something, never state that you did. Put declaration immediately after the report heading and before the reference list, as the brief says; you can also summarise it immediately before references.

## Task 1 (~350 words)

**1.1 [your real short hash]:** Confirm Pool ID from Aula, decision point/target, official source and *applicable* terms. Why pickup features suffice (and cannot forecast payment behaviour perfectly)? Mention cash-only selection and denominator `fare_amount` in the outcome but not in inputs.

**1.2 [hash] five-V + ethics table:** Copy actual figures only from `results/task1.json`: raw/clean rows, monthly records, GiB of downloaded Parquet, per-month schema differences, overlapping validity flags. "Velocity" for this historical archive is monthly volume/publication cadence, not a streaming rate. Check TLC's stated data-accuracy warning. Do not confuse NYC Open Data's 'open by default' policy with a grant of reuse rights for every part of NYC.gov: the site's general Terms of Use include an intellectual-property reservation. Verify and cite the specific dataset terms from your allocation; evaluate payment/geographic risk.

**1.3 [hash] preparation table:** source month normalisation -> invalid-label filtering -> train/Oct/test dates -> FIXED pickup-time category mapping plus within-fold imputation/encoding/scaling -> vector assembly -> model. Explain why learning the imputer/encoder/scaler **inside CV** avoids information leakage, why dropoff, the dictionary's final RatecodeID, fare, tips, total and realised distance are not pickup-time features, and why sine/cosine month features avoid unseen-month indexing, and why date+zone partition hashing can reduce concentration without proving skew is fixed. Take an authentic Task1 screenshot or embed `results/task1_evidence.png` generated on your run.

## Task 2 (~400 words)

**2.1 [hash] selection table:** Four algorithms / explicitly **linear, tree-ensemble and neural** plus GBT free choice; feature width, class-balance caveat and all searched grids from `results/task2.json`. Spark MLP has no weightCol, so its objective is UNWEIGHTED despite sharing the same CV/train rows; do not call it identically class-weighted.

**2.2 [hash] compute table:** For each model: CV fold count, training-only tuning-sample fraction/count, best parameters, actual CV seconds, full Jan–Sep refit seconds, executor/driver settings. What did the best grid values suggest? What does class weighting change? Why hold out October for F0.5 threshold selection?

**2.3 [hash] comparison table:** Report complete Nov–Dec confusion cells and precision, recall, F1, specificity, balanced accuracy, ROC-AUC and PR-AUC for all four. State positive-class prevalence, PR baseline and *the predeclared CV selection rule* rather than selecting on the final test. Trade off model cost against benefit; note CV is day-grouped but not forward chaining. Embed the actual Task2 figure (`task2_evidence.png`).

## Task 3 (~300 words)

**3.1 [hash] Spark UI/optimisation table:** Record **real** stage/job ID, stage wall time, shuffle read/write, spill, task-duration distribution and your interpretation. Pair each measured change with same-query before/after, repeat count and cache fill. AQE's skew-join rule does not prove skew in a groupBy.

**3.2 [hash] official test perturbation table:** From `results/task3.json`, quote `test_perturbation_protocol` and show all FOUR saved models' signed `Δ F1`, `Δ AUC-ROC` and rank on the SAME frozen perturbed Nov–Dec test cohort. The October thresholds and labels do not change. Optional `retraining_stability_supplement` is a separate train-subsampling check and does NOT replace the required test-data perturbation. Discuss why one synthetic input-error scenario cannot prove deployment reliability.

**3.3 [hash] explanation/fairness:** Embed the observed LIME figure, five features, **local** surrogate R²; don't interpret local LIME weights as a global importance ranking. Report borough-specific observed base rates vs predicted positive rates and FPR/TPR with group n, name the geographic/payment-selection bias mechanisms. Propose an ablation/calibration check; say why geography is not itself a protected attribute.

## Task 4 (~150 words)

**4.1 [hash]:** Embed a legible **real** four-dashboard contact sheet and link the live workbook. State design choice and safe aggregate-data export rule.

**4.2 [hash]:** Three insights written as **observation -> measured evidence (dashboard #) -> decision**. Critically reflect on observed cash-label limits, temporal shift/monitoring after deployment, local explanation limits and real compute costs. Explain one specific next test or serving-time safeguard rather than generic improvement claims.

## References

Verify URLs and bibliographic details yourself: official TLC data/dictionary/terms/zone lookup; Apache Spark version-specific ML and evaluator docs; LIME original method if cited; trustworthy fairness references. Cite sources in prose and list them consistently. DO NOT invent papers or present the module brief on a website.
