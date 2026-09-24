"""Test-only synthetic Word assembly; NEVER produces a coursework report.

All fixtures live in pytest's temporary directory. Git/URL checks are mocked
ONLY to exercise document layout; the real builder fails closed without them.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from docx import Document
from PIL import Image

from coursework.settings import write_json
from scripts import build_report


def test_report_refuses_missing_real_evidence(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises((KeyError, FileNotFoundError, ValueError)):
        build_report.build(str(tmp_path / "config.json"), str(tmp_path / "author.json"), str(tmp_path / "draft.docx"))
    assert not (tmp_path / "draft.docx").exists()


def test_synthetic_docx_assembly_in_temporary_directory(tmp_path, monkeypatch):
    resultdir = tmp_path / "results"; resultdir.mkdir()
    cfg = {"student": {"name": "Synthetic Tester", "email": "tester@coventry.ac.uk", "sid": "1234567"},
           "allocation": {"confirmed_against_aula_register": True, "pool_reference": "TR-04", "licence_checked": True,
                          "licence_or_terms_url": "https://example.org/test-only"},
           "paths": {"results_dir": str(resultdir)}}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg))
    one = {"status": "observed", "pool_reference": "TR-04", "source_page": "https://example.org/data",
           "licence_or_terms_url": "https://example.org/terms", "raw_row_count": 20_000_000,
           "file_size_gib": 2.0, "raw_rows_by_source_month": {"1": 1_000_000, "2": 2_000_000},
           "raw_schema_by_month": [{"original_column_count": 19}, {"original_column_count": 19}],
           "quality_counts_overlapping": {"invalid_fare": 100, "not_credit_card": 2000},
           "preprocessing_stages": ["Imputer", "StringIndexer", "OneHotEncoder", "VectorAssembler", "StandardScaler"]}
    names = ["LogisticRegression", "RandomForestClassifier", "MultilayerPerceptronClassifier", "GBTClassifier"]
    models = []
    families = ["linear", "tree_ensemble", "neural", "free_choice_boosted_tree"]
    for name, family in zip(names, families):
        models.append({"name": name, "family": family, "param_grid": [{"regParam": 0.1}],
                       "best_params": {"regParam": 0.1}, "cv_folds": 3, "cv_seconds": 10.0,
                       "final_fit_seconds": 20.0,
                       "metrics": {"accuracy": 0.8, "balanced_accuracy": 0.7, "specificity": 0.6,
                                   "auc_roc": 0.8, "auc_pr": 0.75, "base_rate": 0.70, "positive_precision": 0.7,
                                   "positive_recall": 0.8, "positive_f1": 0.75,
                                   "confusion": {"tn": 10, "fp": 2, "fn": 1, "tp": 20}}})
    two = {"status": "observed", "models": models, "recommended_model": "LogisticRegression"}
    stab = {n: {"baseline_metrics": {"auc_pr": 0.75},
                "signed_deltas": {"auc_pr": 0.01, "auc_roc": 0.02, "positive_f1": -0.01},
                "rank": rank} for rank, n in enumerate(names, 1)}
    three = {"status": "observed",
        "test_perturbation_protocol": {"perturbation": "synthetic-only frozen test inputs",
                                       "labels_changed": 0},
        "optimisations": [
        {"change": "shuffle_partition_count", "partitions": 8, "median_seconds": 12.0},
        {"change": "persist_MEMORY_AND_DISK", "median_before_seconds": 12.0,
         "median_after_seconds": 7.0, "cache_fill_seconds": 3.0}], "stability": stab,
        "explainability": {"feature_weights": [{"feature": f"fake_feature_{i}", "local_weight": 0.1}
                                                for i in range(5)]}}
    four = {"status": "observed", "tableau_public_link": "https://public.tableau.com/views/test/d1",
            "link_check": {"valid": True}, "dashboards": ["d1", "d2", "d3", "d4"]}
    for i, payload in enumerate((one, two, three, four), 1):
        write_json(resultdir / f"task{i}.json", payload)
    for name in build_report.FIGURES.values():
        Image.new("RGB", (320, 180), "white").save(resultdir / name)
    # Repetitions contain exactly the student's test-only AUTHOR content; nothing
    # from this fixture is saved in the deliverable's real results/report directory.
    original = ("I examined this synthetic fixture and checked its fields "
                "against the same held out cohort. I would not infer a causal "
                "effect from the observed association. The actual coursework "
                "needs its own measured results, institutional context, source "
                "terms, and an honestly documented limitation for deployment. ")
    author = {"repository_url": "https://github.com/7006SCN2627SEPNOV/7006SCN_AB_1234567",
              "branch": "main",
              "ai_use_declaration": "For this synthetic test I used automation to assemble figures and tables; "
                    "the author must disclose any actual tools used, their specific sections, "
                    "and every step personally reviewed and checked before submission.",
              "sections": {sid: {"text": original * (2 if i < 8 else 1), "commit": "abcdef0"}
                           for i, sid in enumerate(build_report.SECTION_IDS)},
              "references": ["Synthetic testing reference, not for assessment"]}
    author_file = tmp_path / "author.json"; author_file.write_text(json.dumps(author))
    monkeypatch.setattr(build_report, "git_provenance", lambda *args: None)
    monkeypatch.setattr(build_report, "verify_public_link", lambda *args: {"valid": True})
    target = tmp_path / "synthetic-only.docx"
    try:
        path = build_report.build(str(config_path), str(author_file), str(target))
    except ValueError as exc:
        # Word count failure is a real blocker, not a pass for full assembly.
        pytest.fail(f"Expected a valid synthetic DOCX, got: {exc}")
    else:
        assert path.is_file()
        doc = Document(path)
        assert "AI Use Declaration" in [p.text for p in doc.paragraphs]
        assert any("Word count" in p.text for p in doc.paragraphs)
        assert len(doc.inline_shapes) == 5
        assert len(doc.part.rels) >= 5  # actual hyperlink relationships created
