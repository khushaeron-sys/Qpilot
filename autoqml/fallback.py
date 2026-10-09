"""Deterministic plan used when the LLM fails or the API is down. Owner: Agent dev."""
import math
import re
from typing import Optional
from .catalog import MIN_QUBITS, MAX_QUBITS
from .schemas import AnsatzCfg, DatasetProfile, FeatureMapCfg, OptimizerCfg, QuantumPlan


def parse_qubit_cap(prompt: str) -> Optional[int]:
    m = re.search(r"(\d+)\s*[- ]?qubit", prompt, re.I)
    return int(m.group(1)) if m else None


def default_plan(profile: DatasetProfile, prompt: str = "") -> QuantumPlan:
    cap = parse_qubit_cap(prompt) or 4
    need = max(MIN_QUBITS, math.ceil(math.log2(max(profile.n_classes, 2))))
    n = max(need, min(4, cap, profile.n_features_encoded, MAX_QUBITS))
    return QuantumPlan(
        rationale="Deterministic fallback: small, fast, well-understood PQC configuration.",
        task_type="binary_classification" if profile.n_classes == 2 else "multiclass_classification",
        n_qubits=n,
        dim_reduction="none" if n == profile.n_features_encoded else "pca",
        feature_map=FeatureMapCfg(name="ZFeatureMap", reps=1, entanglement=None),
        ansatz=AnsatzCfg(name="EfficientSU2", reps=2, entanglement="linear"),
        optimizer=OptimizerCfg(name="COBYLA", maxiter=100),
        train_subsample=120, classical_baseline="logistic_regression", seed=42,
    )
