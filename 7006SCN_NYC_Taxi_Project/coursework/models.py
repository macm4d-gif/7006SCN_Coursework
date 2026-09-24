"""Actual Spark ML pipelines and defensible, bounded 4-model CV search grids."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    GBTClassifier, LogisticRegression, MultilayerPerceptronClassifier,
    RandomForestClassifier,
)
from pyspark.ml.feature import Imputer, OneHotEncoder, StandardScaler, StringIndexerModel, VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder

from .data import CATEGORICAL_FEATURES, FORBIDDEN_AT_PICKUP, NUMERIC_FEATURES

# Public, predeclared feature domains: using a fixed vocabulary before fitting
# avoids inspecting November--December to design the feature vector and gives
# EVERY CV fold exactly the same input width for the neural network. All other
# values fall into StringIndexerModel's handleInvalid='keep' category.
PREDECLARED_CATEGORIES = {
    "pickup_borough": ("Bronx", "Brooklyn", "EWR", "Manhattan", "Queens", "Staten Island", "Unknown"),
    "VendorID": ("1", "2", "4", "Unknown"),
    "pickup_hour": tuple(str(hour) for hour in range(24)),
    "pickup_dow": tuple(str(day) for day in range(1, 8)),
}
assert set(PREDECLARED_CATEGORIES) == set(CATEGORICAL_FEATURES)
# OneHotEncoder(handleInvalid='keep', dropLast=False) adds a separate extra
# invalid-index position to StringIndexerModel's fixed known + unknown labels.
FEATURE_VECTOR_SIZE = len(NUMERIC_FEATURES) + sum(
    len(PREDECLARED_CATEGORIES[col]) + 2 for col in CATEGORICAL_FEATURES
)


@dataclass
class ModelSpec:
    name: str
    family: str
    estimator: Any
    param_grid: list[dict]
    rationale_to_check: str


def make_preprocessing_pipeline(*, omit_borough: bool = False) -> Pipeline:
    """FIT learned imputation, OHE metadata and scaling INSIDE each CV fold.

    Categorical mappings are fixed from documented pickup-time domains, not
    learned on the final-test rows. They make MLP input width fold-invariant.
    """
    categorical = [x for x in CATEGORICAL_FEATURES if not (omit_borough and x == "pickup_borough")]
    actual_inputs = set(categorical) | set(NUMERIC_FEATURES)
    assert actual_inputs.isdisjoint(FORBIDDEN_AT_PICKUP), "A post-pickup/target-derived field entered the feature vector."
    idx = [StringIndexerModel.from_labels(list(PREDECLARED_CATEGORIES[col]),
           inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep")
           for col in categorical]
    encode = OneHotEncoder(
        inputCols=[f"{c}_idx" for c in categorical],
        outputCols=[f"{c}_onehot" for c in categorical],
        handleInvalid="keep", dropLast=False,
    )
    stages = [Imputer(inputCols=list(NUMERIC_FEATURES), outputCols=[f"{c}_imputed" for c in NUMERIC_FEATURES], strategy="median")]
    stages.extend(idx)
    stages.extend([
        encode,
        VectorAssembler(inputCols=[f"{x}_imputed" for x in NUMERIC_FEATURES] +
                        [f"{x}_onehot" for x in categorical], outputCol="assembled", handleInvalid="error"),
        # withMean=False preserves one-hot sparsity; trees do not need scaling but
        # a shared representation makes the model-family comparison controlled.
        StandardScaler(inputCol="assembled", outputCol="features", withStd=True, withMean=False),
    ])
    return Pipeline(stages=stages)


def pipeline_for(estimator, *, omit_borough: bool = False) -> Pipeline:
    return Pipeline(stages=make_preprocessing_pipeline(omit_borough=omit_borough).getStages() + [estimator])


def build_model_specs(*, smoke: bool = False) -> list[ModelSpec]:
    """All four models are tuned by CrossValidator in Task2; smoke=True is TEST ONLY."""
    lr = LogisticRegression(featuresCol="features", labelCol="label", weightCol="class_weight", maxIter=80)
    rf = RandomForestClassifier(featuresCol="features", labelCol="label", weightCol="class_weight", seed=31,
                                maxBins=64, featureSubsetStrategy="sqrt")
    # Spark MLP has NO weightCol in 3.5.x. The unweighted objective is disclosed
    # per model; all four still see the same CV sample and Jan--Sep final rows.
    mlp = MultilayerPerceptronClassifier(featuresCol="features", labelCol="label",
        layers=[FEATURE_VECTOR_SIZE, 8, 2], maxIter=25, blockSize=256, seed=31)
    gbt = GBTClassifier(featuresCol="features", labelCol="label", weightCol="class_weight", seed=31, maxBins=64)
    lr_grid = ParamGridBuilder().addGrid(lr.regParam, [0.005, 0.05]).addGrid(lr.elasticNetParam, [0.0, 0.5]).build()
    rf_grid = ParamGridBuilder().addGrid(rf.numTrees, [40, 80]).addGrid(rf.maxDepth, [7, 11]).build()
    mlp_grid = ParamGridBuilder().addGrid(mlp.layers,
        [[FEATURE_VECTOR_SIZE, 8, 2], [FEATURE_VECTOR_SIZE, 16, 2]]).addGrid(mlp.maxIter, [20, 40]).build()
    gbt_grid = ParamGridBuilder().addGrid(gbt.maxDepth, [4, 6]).addGrid(gbt.maxIter, [20, 40]).build()
    specs = [
        ModelSpec("LogisticRegression", "linear", lr, lr_grid, "Sparse, interpretable baseline; regularisation handles correlated time/borough features."),
        ModelSpec("RandomForestClassifier", "tree_ensemble", rf, rf_grid, "Nonlinear borough-by-hour interactions; bagging can reduce variance."),
        ModelSpec("MultilayerPerceptronClassifier", "neural", mlp, mlp_grid,
                  "Nonlinear neural comparator with fixed-width inputs. Spark 3.5 MLP cannot use sample weights; evaluate unweighted training and label imbalance honestly."),
        ModelSpec("GBTClassifier", "free_choice_boosted_tree", gbt, gbt_grid, "Test whether boosting beats bagging at a potentially higher training cost."),
    ]
    if smoke:
        # Tiny, explicitly unassessable grids keep CI/integration tests short.
        lightweight = {
            "LogisticRegression": {"regParam": 0.05, "elasticNetParam": 0.0},
            "RandomForestClassifier": {"numTrees": 3, "maxDepth": 2},
            "MultilayerPerceptronClassifier": {"layers": [FEATURE_VECTOR_SIZE, 4, 2], "maxIter": 4},
            "GBTClassifier": {"maxDepth": 2, "maxIter": 2},
        }
        for spec in specs:
            spec.param_grid = [{spec.estimator.getParam(key): value
                                for key, value in lightweight[spec.name].items()}]
    return specs


def param_grid_as_dicts(spec: ModelSpec) -> list[dict]:
    return [{p.name: v for p, v in one.items()} for one in spec.param_grid]


def use_best_params(spec: ModelSpec, chosen: dict) -> Any:
    """Create a fresh classifier; avoids reusing the fold-fitted CV bestModel."""
    changes = {spec.estimator.getParam(name): value for name, value in chosen.items()}
    return spec.estimator.copy(changes)


def tree_feature_importance(fitted_pipeline, one_row) -> list[dict]:
    """Tree ensemble's GLOBAL split importances, aggregated over one-hot columns.

    Gets the assembler's real trained attribute names BEFORE StandardScaler strips
    them. Zero-value features are retained; importances are model-dependent, not
    causal effect sizes or comparable to a local LIME weight.
    """
    from collections import defaultdict
    from pyspark.ml import PipelineModel

    model = fitted_pipeline.stages[-1]
    if not hasattr(model, "featureImportances"):
        return []
    # Last stages are VectorAssembler -> StandardScalerModel -> classifier model.
    before_scaling = PipelineModel(stages=fitted_pipeline.stages[:-2]).transform(one_row.limit(1))
    attributes = before_scaling.schema["assembled"].metadata["ml_attr"].get("attrs", {})
    by_index = {int(x["idx"]): x["name"] for group in attributes.values() for x in group}
    result = defaultdict(float)
    for idx, value in enumerate(model.featureImportances.toArray()):
        name = by_index.get(idx, f"unknown_vector_index_{idx}")
        original = next((key for key in (*CATEGORICAL_FEATURES, *NUMERIC_FEATURES)
                         if name.startswith(key + "_")), name)
        if original in ("pickup_month_sin", "pickup_month_cos"):
            original = "pickup_month_cyclic"
        result[original] += float(value)
    return [{"feature": key, "global_split_importance": value}
            for key, value in sorted(result.items(), key=lambda pair: -pair[1])]
