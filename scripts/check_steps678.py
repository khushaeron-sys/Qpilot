"""
Test Steps 6-7-8: VQC trainer, baseline, evaluator.
Run with: .venv\Scripts\python scripts\check_steps678.py
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from autoqml.preprocess import prepare_data
from autoqml.schemas import QuantumPlan, FeatureMapCfg, AnsatzCfg, OptimizerCfg
from autoqml.quantum.trainer import train_qnn
from autoqml.baseline import run_baselines
from autoqml.evaluator import evaluate

df = pd.read_csv(Path(__file__).parents[1] / "data" / "breast_cancer.csv")
print("Loaded:", df.shape)

plan = QuantumPlan(
    rationale="step-6 test",
    task_type="binary_classification",
    n_qubits=4,
    dim_reduction="pca",
    feature_map=FeatureMapCfg(name="ZFeatureMap", reps=1, entanglement=None),
    ansatz=AnsatzCfg(name="EfficientSU2", reps=2, entanglement="linear"),
    optimizer=OptimizerCfg(name="COBYLA", maxiter=30),
    train_subsample=80,
    classical_baseline="logistic_regression",
    seed=42,
)

data = prepare_data(df, "target", plan)
print("Data prepared: X_train_q=%s  X_test_q=%s" % (data.X_train_q.shape, data.X_test_q.shape))

print("Training VQC (COBYLA maxiter=30, 4 qubits, ~30s)...")
qnn = train_qnn(plan, data, sampler_kind="reference")

print("\n=== STEP 6 — QNN Result ===")
print("  train_accuracy  : %.4f" % qnn.train_accuracy)
print("  test_accuracy   : %.4f" % qnn.test_accuracy)
print("  test_f1_macro   : %.4f" % qnn.test_f1_macro)
print("  loss_pts        : %d" % len(qnn.loss_history))
if qnn.loss_history:
    print("  loss[0]         : %.4f" % qnn.loss_history[0])
    print("  loss[-1]        : %.4f" % qnn.loss_history[-1])
print("  n_params        : %d" % qnn.n_params)
print("  circuit_depth   : %d" % qnn.circuit_depth)
print("  cx_count        : %d" % qnn.cx_count)
print("  train_seconds   : %.1fs" % qnn.train_seconds)

assert 0.0 <= qnn.test_accuracy <= 1.0, "QNN accuracy out of range"
assert qnn.test_accuracy > 0.5, "QNN accuracy too low: %.4f" % qnn.test_accuracy
assert len(qnn.loss_history) > 0, "No loss history — callback/tracking broken"
assert qnn.n_params > 0, "Zero trainable params"
print("Step 6 (trainer): PASS")

base = run_baselines(data, plan)
print("\n=== STEP 7 — Baseline Result ===")
print("  name              : %s" % base.name)
print("  matched_accuracy  : %.4f" % base.matched_accuracy)
print("  matched_f1_macro  : %.4f" % base.matched_f1_macro)
print("  full_accuracy     : %.4f" % base.full_accuracy)
print("  train_seconds     : %.1fs" % base.train_seconds)
assert 0.0 <= base.matched_accuracy <= 1.0
print("Step 7 (baseline): PASS")

ev = evaluate(qnn, base)
print("\n=== STEP 8 — Evaluation ===")
print("  delta_vs_matched  : %+.4f" % ev.delta_vs_matched)
print("  delta_vs_full     : %+.4f" % ev.delta_vs_full)
print("  overfit_gap       : %+.4f" % ev.overfit_gap)
print("  loss_trend        : %s" % ev.loss_trend)
print("  verdict           : %s" % ev.verdict)
print("Step 8 (evaluator): PASS")

print("\n" + "="*55)
print("All Steps 6-7-8: PASS")
print("="*55)
