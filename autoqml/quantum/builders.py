"""Catalog -> Qiskit 1.x objects. The ONLY place the LLM's strings become code. Owner: Quantum dev."""
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import EfficientSU2, RealAmplitudes, ZFeatureMap, ZZFeatureMap
from qiskit_algorithms.optimizers import COBYLA, SPSA

from ..schemas import AnsatzCfg, FeatureMapCfg, OptimizerCfg, QuantumPlan


def build_feature_map(cfg: FeatureMapCfg, n: int) -> QuantumCircuit:
    if cfg.name == "ZFeatureMap":
        return ZFeatureMap(feature_dimension=n, reps=cfg.reps)
    if cfg.name == "ZZFeatureMap":
        return ZZFeatureMap(feature_dimension=n, reps=cfg.reps, entanglement=cfg.entanglement or "full")
    raise ValueError(f"Feature map not in catalog: {cfg.name}")


def build_ansatz(cfg: AnsatzCfg, n: int) -> QuantumCircuit:
    if cfg.name == "RealAmplitudes":
        return RealAmplitudes(num_qubits=n, reps=cfg.reps, entanglement=cfg.entanglement)
    if cfg.name == "EfficientSU2":
        return EfficientSU2(num_qubits=n, reps=cfg.reps, entanglement=cfg.entanglement)
    raise ValueError(f"Ansatz not in catalog: {cfg.name}")


def build_optimizer(cfg: OptimizerCfg):
    if cfg.name == "COBYLA":
        return COBYLA(maxiter=cfg.maxiter)
    if cfg.name == "SPSA":
        return SPSA(maxiter=cfg.maxiter)
    raise ValueError(f"Optimizer not in catalog: {cfg.name}")


def build_circuit(plan: QuantumPlan) -> QuantumCircuit:
    """Full PQC = feature map (data encoding) + ansatz (trainable). Used for display & stats."""
    qc = QuantumCircuit(plan.n_qubits)
    qc.compose(build_feature_map(plan.feature_map, plan.n_qubits), inplace=True)
    qc.compose(build_ansatz(plan.ansatz, plan.n_qubits), inplace=True)
    return qc


def circuit_stats(plan: QuantumPlan) -> dict:
    t = transpile(build_circuit(plan), basis_gates=["u", "cx"], optimization_level=0)
    return {"depth": int(t.depth()), "cx_count": int(t.count_ops().get("cx", 0)),
            "n_params": int(build_ansatz(plan.ansatz, plan.n_qubits).num_parameters)}
