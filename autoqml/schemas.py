"""Shared data contracts. Owner: Agent/Backend dev. FREEZE THESE BY HOUR 2."""
from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from .catalog import (AnsatzName, BaselineName, DimReductionName, Entanglement,
                      FeatureMapName, OptimizerName, MAX_MAXITER, MAX_QUBITS,
                      MAX_TRAIN_SAMPLES, MIN_QUBITS)


# ───────────── Input ─────────────
class RunRequest(BaseModel):
    user_prompt: str
    target_column: str


class ColumnInfo(BaseModel):
    name: str
    kind: Literal["numeric", "categorical", "boolean"]
    n_unique: int
    n_missing: int


class DatasetProfile(BaseModel):
    n_rows: int
    n_features_raw: int
    n_features_encoded: int          # after one-hot + dropping constants/IDs
    target_column: str
    n_classes: int
    class_balance: Dict[str, float]
    missing_pct: float
    columns: List[ColumnInfo]


# ───────────── Planner output (what the LLM is allowed to produce) ─────────────
class FeatureMapCfg(BaseModel):
    name: FeatureMapName
    reps: int = Field(2, ge=1, le=3)
    entanglement: Optional[Entanglement] = Field(None, description="Only for ZZFeatureMap; null for ZFeatureMap")


class AnsatzCfg(BaseModel):
    name: AnsatzName
    reps: int = Field(2, ge=1, le=4)
    entanglement: Entanglement = "linear"


class OptimizerCfg(BaseModel):
    name: OptimizerName
    maxiter: int = Field(100, ge=10, le=MAX_MAXITER)


class QuantumPlan(BaseModel):
    # NOTE: `rationale` is FIRST on purpose: with a forced tool call the model has no
    # free-text scratchpad, so the first field acts as a short chain-of-thought.
    rationale: str = Field(..., max_length=700, description="2-4 sentences justifying the choices")
    task_type: Literal["binary_classification", "multiclass_classification"]
    n_qubits: int = Field(..., ge=MIN_QUBITS, le=MAX_QUBITS)
    dim_reduction: DimReductionName
    feature_map: FeatureMapCfg
    ansatz: AnsatzCfg
    optimizer: OptimizerCfg
    train_subsample: int = Field(150, ge=50, le=MAX_TRAIN_SAMPLES)
    classical_baseline: BaselineName = "logistic_regression"
    seed: int = 42


class Review(BaseModel):
    """Critic output after seeing results."""
    diagnosis: str = Field(..., max_length=500)
    action: Literal["accept", "revise"]
    revised_plan: Optional[QuantumPlan] = None


# ───────────── Results ─────────────
class QNNResult(BaseModel):
    train_accuracy: float
    test_accuracy: float
    test_f1_macro: float
    loss_history: List[float]        # one entry per objective evaluation
    n_params: int
    n_qubits: int
    circuit_depth: int               # after transpile to [u, cx]
    cx_count: int
    train_seconds: float
    weights: List[float]


class BaselineResult(BaseModel):
    name: str
    matched_accuracy: float          # same reduced features + same train rows as the QNN
    matched_f1_macro: float
    matched_n_features: int
    full_accuracy: float             # all features, all train rows (best-effort classical)
    full_f1_macro: float
    full_n_features: int
    train_seconds: float


class Evaluation(BaseModel):
    delta_vs_matched: float          # QNN - matched baseline (accuracy points, 0.03 = +3 pts)
    delta_vs_full: float
    overfit_gap: float               # train_acc - test_acc
    loss_trend: Literal["still_decreasing", "plateaued", "noisy", "too_short"]
    verdict: str


class Attempt(BaseModel):
    plan: QuantumPlan
    qnn: QNNResult
    baseline: BaselineResult
    evaluation: Evaluation
    critic_note: Optional[str] = None


class RunReport(BaseModel):
    request: RunRequest
    profile: DatasetProfile
    planner_meta: Dict[str, Any]
    attempts: List[Attempt]
