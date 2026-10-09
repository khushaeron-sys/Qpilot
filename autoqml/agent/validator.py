"""Hard guardrails in CODE (the prompt is advice, this is law). Owner: Agent dev."""
import math
from typing import List, Optional
from ..catalog import MAX_CLASSES, MAX_PARAMS, n_ansatz_params
from ..schemas import DatasetProfile, QuantumPlan


def validate_plan(plan: QuantumPlan, profile: DatasetProfile, qubit_cap: Optional[int] = None) -> List[str]:
    errs: List[str] = []
    if profile.n_classes < 2 or profile.n_classes > MAX_CLASSES:
        errs.append(f"dataset has {profile.n_classes} classes; supported range is 2..{MAX_CLASSES}")
    want_type = "binary_classification" if profile.n_classes == 2 else "multiclass_classification"
    if plan.task_type != want_type:
        errs.append(f"task_type must be '{want_type}' for {profile.n_classes} classes")
    if plan.n_qubits > profile.n_features_encoded:
        errs.append(f"n_qubits={plan.n_qubits} exceeds available features ({profile.n_features_encoded}); "
                    f"each qubit encodes exactly one feature")
    if plan.n_qubits < math.ceil(math.log2(max(profile.n_classes, 2))):
        errs.append(f"n_qubits too small to represent {profile.n_classes} classes")
    if qubit_cap is not None and plan.n_qubits > qubit_cap:
        errs.append(f"user capped qubits at {qubit_cap} but plan uses {plan.n_qubits}")
    if plan.dim_reduction == "none" and plan.n_qubits != profile.n_features_encoded:
        errs.append(f"dim_reduction='none' requires n_qubits == n_features_encoded ({profile.n_features_encoded}); "
                    f"use 'pca' or 'select_k_best'")
    if plan.dim_reduction != "none" and plan.n_qubits >= profile.n_features_encoded:
        errs.append("dim_reduction must be 'none' when n_qubits equals the number of features")
    if plan.feature_map.name == "ZFeatureMap" and plan.feature_map.entanglement is not None:
        plan.feature_map.entanglement = None          # harmless: normalise instead of failing
    if plan.feature_map.name == "ZZFeatureMap" and plan.feature_map.entanglement is None:
        plan.feature_map.entanglement = "linear"
    n_par = n_ansatz_params(plan.ansatz.name, plan.n_qubits, plan.ansatz.reps)
    if n_par > MAX_PARAMS:
        errs.append(f"ansatz has {n_par} trainable parameters (max {MAX_PARAMS}); reduce reps/qubits or use RealAmplitudes")
    if plan.train_subsample > profile.n_rows * 0.75:
        plan.train_subsample = max(50, int(profile.n_rows * 0.75))   # normalise
    return errs
