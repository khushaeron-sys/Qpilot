"""
Smoke test for the full AutoQML pipeline (Step 15).
Runs end-to-end on breast_cancer.csv with a FAKE LLM call (no API key needed).
Asserts both accuracies are between 0 and 1.
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from autoqml.agent.orchestrator import run_autoqml
from autoqml.schemas import RunRequest

# ---------------------------------------------------------------------------
# Fake LLM that immediately returns a valid 4-qubit plan (no API key needed)
# ---------------------------------------------------------------------------
VALID_PLAN = {
    "rationale": "Smoke-test plan: 4 qubits, PCA, ZFeatureMap, EfficientSU2, COBYLA.",
    "task_type": "binary_classification",
    "n_qubits": 4,
    "dim_reduction": "pca",
    "feature_map": {"name": "ZFeatureMap", "reps": 1, "entanglement": None},
    "ansatz": {"name": "EfficientSU2", "reps": 2, "entanglement": "linear"},
    "optimizer": {"name": "COBYLA", "maxiter": 30},
    "train_subsample": 80,
    "classical_baseline": "logistic_regression",
    "seed": 42,
}

CRITIC_ACCEPT = {
    "diagnosis": "Smoke test: accepting immediately.",
    "action": "accept",
    "revised_plan": None,
}

_call_counter = 0


def fake_call(system, messages, tool, max_tokens=1500):
    """Return the plan on first call, accept on critic call."""
    global _call_counter
    _call_counter += 1
    if tool["name"] == "submit_quantum_plan":
        payload = VALID_PLAN
    else:
        payload = CRITIC_ACCEPT
    tool_id = f"toolu_smoke_{_call_counter}"
    content_block = [{"type": "tool_use", "id": tool_id, "name": tool["name"], "input": payload}]
    return tool_id, payload, content_block


def test_pipeline_breast_cancer():
    global _call_counter
    _call_counter = 0

    data_path = Path(__file__).resolve().parents[1] / "data" / "breast_cancer.csv"
    assert data_path.exists(), f"breast_cancer.csv not found at {data_path}"

    df = pd.read_csv(data_path)
    print(f"\nLoaded breast_cancer.csv: {df.shape[0]} rows x {df.shape[1]} cols")

    request = RunRequest(
        user_prompt="Classify malignant vs benign. Prioritize accuracy, keep under 6 qubits.",
        target_column="target",
    )

    events = []

    def on_event(kind, payload):
        events.append(kind)
        if kind == "profile":
            p = payload
            print(f"  [profile] {p.n_rows} rows, {p.n_features_encoded} features, {p.n_classes} classes")
        elif kind == "plan":
            plan, meta = payload
            print(f"  [plan] source={meta['source']}, {plan.n_qubits} qubits, {plan.dim_reduction}, "
                  f"{plan.feature_map.name}+{plan.ansatz.name}(r{plan.ansatz.reps}), "
                  f"{plan.optimizer.name}({plan.optimizer.maxiter})")
        elif kind == "training":
            print("  [training] fitting VQC on simulator...")
        elif kind == "trained":
            att = payload
            print(f"  [trained] QNN test_acc={att.qnn.test_accuracy:.3f}, "
                  f"classical_matched={att.baseline.matched_accuracy:.3f}, "
                  f"time={att.qnn.train_seconds:.1f}s, "
                  f"loss_pts={len(att.qnn.loss_history)}")
            print(f"  [verdict] {att.evaluation.verdict}")
        elif kind == "critic":
            print(f"  [critic] {payload}")

    report = run_autoqml(
        df,
        request,
        max_iterations=1,
        sampler_kind="reference",
        on_event=on_event,
        call=fake_call,
    )

    att = report.attempts[0]
    q_acc = att.qnn.test_accuracy
    c_acc = att.baseline.matched_accuracy

    print(f"\n{'=' * 60}")
    print(f"QNN  test accuracy  : {q_acc:.4f}")
    print(f"Classical (matched) : {c_acc:.4f}")
    print(f"Circuit depth       : {att.qnn.circuit_depth}")
    print(f"CX count            : {att.qnn.cx_count}")
    print(f"n_params            : {att.qnn.n_params}")
    print(f"Train seconds       : {att.qnn.train_seconds:.1f}s")
    print(f"Loss history pts    : {len(att.qnn.loss_history)}")
    print(f"Loss trend          : {att.evaluation.loss_trend}")
    print(f"Verdict             : {att.evaluation.verdict}")
    print(f"{'=' * 60}")

    assert 0.0 <= q_acc <= 1.0, f"QNN accuracy out of range: {q_acc}"
    assert 0.0 <= c_acc <= 1.0, f"Classical accuracy out of range: {c_acc}"
    assert q_acc > 0.5, f"QNN accuracy suspiciously low: {q_acc}"
    assert len(att.qnn.loss_history) > 0, "No loss history recorded"
    assert att.qnn.n_params > 0, "Zero trainable parameters"
    print("\nAll assertions passed - pipeline smoke test OK")
    return report


if __name__ == "__main__":
    test_pipeline_breast_cancer()
