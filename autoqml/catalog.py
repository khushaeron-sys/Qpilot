"""Single source of truth for everything the LLM is allowed to choose. Owner: Quantum dev."""
from typing import Literal

# ---- hard limits (calibrate MAX_* after the hour-4 benchmark) ----
MIN_QUBITS, MAX_QUBITS = 2, 8
MAX_MAXITER = 200
MAX_TRAIN_SAMPLES = 300
MAX_PARAMS = 64
MAX_CLASSES = 4

# ---- allowed Qiskit objects (Qiskit 1.x class-based circuit library) ----
FeatureMapName = Literal["ZFeatureMap", "ZZFeatureMap"]
AnsatzName = Literal["RealAmplitudes", "EfficientSU2"]
Entanglement = Literal["linear", "circular", "full"]
OptimizerName = Literal["COBYLA", "SPSA"]
DimReductionName = Literal["none", "pca", "select_k_best"]
BaselineName = Literal["logistic_regression", "random_forest", "mlp"]


def n_ansatz_params(name: str, n_qubits: int, reps: int) -> int:
    """RealAmplitudes: n*(reps+1). EfficientSU2: 2*n*(reps+1)."""
    per_layer = {"RealAmplitudes": 1, "EfficientSU2": 2}[name]
    return per_layer * n_qubits * (reps + 1)
