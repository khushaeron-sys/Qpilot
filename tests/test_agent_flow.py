import json, os, warnings
warnings.filterwarnings("ignore")
import pandas as pd
from autoqml.agent.llm import tool_schema
from autoqml.agent.orchestrator import run_autoqml
from autoqml.agent.planner import PLAN_TOOL
from autoqml.agent.critic import REVIEW_TOOL
from autoqml.schemas import RunRequest

good = {"rationale": "x", "task_type": "binary_classification", "n_qubits": 4, "dim_reduction": "pca",
        "feature_map": {"name": "ZFeatureMap", "reps": 1, "entanglement": None},
        "ansatz": {"name": "EfficientSU2", "reps": 2, "entanglement": "linear"},
        "optimizer": {"name": "COBYLA", "maxiter": 100}, "train_subsample": 120,
        "classical_baseline": "logistic_regression", "seed": 42}
bad = dict(good, n_qubits=12)                         # violates schema (le=8) AND the user's cap
over_cap = dict(good, n_qubits=7)                     # schema-valid but breaks user's "under 5 qubits"

script = iter([
    ("planner", bad), ("planner", over_cap), ("planner", good),
    ("critic", {"diagnosis": "Underfit: loss still falling.", "action": "revise",
                "revised_plan": dict(good, optimizer={"name": "COBYLA", "maxiter": 150})}),
])
log = []
def fake_call(system, messages, tool):
    kind, payload = next(script)
    log.append((tool["name"], len(messages)))
    return f"toolu_{len(log)}", payload, [{"type": "tool_use", "id": f"toolu_{len(log)}", "name": tool["name"], "input": payload}]

# schemas are plain, ref-free JSON
for t in (PLAN_TOOL, REVIEW_TOOL):
    s = json.dumps(t["input_schema"]); assert "$ref" not in s and "$defs" not in s
print("tool schemas OK; plan props:", list(PLAN_TOOL["input_schema"]["properties"])[:4], "...")

df = pd.read_csv("data/breast_cancer.csv")
rep = run_autoqml(df, RunRequest(user_prompt="Predict malignancy, accuracy first, under 5 qubits", target_column="target"),
                  max_iterations=2, on_event=lambda k, p: print("  event:", k), call=fake_call)
print("planner_meta:", rep.planner_meta)
print("calls (tool, n_messages):", log)
print("attempts:", len(rep.attempts), [round(a.qnn.test_accuracy, 2) for a in rep.attempts], "| maxiter:", [a.plan.optimizer.maxiter for a in rep.attempts])
assert rep.planner_meta["source"] == "claude" and rep.planner_meta["attempts"] == 3

# No API key / API down -> fallback path
def boom(*a, **k): raise RuntimeError("no key")
iris = pd.read_csv("data/iris.csv")
rep2 = run_autoqml(iris, RunRequest(user_prompt="classify species", target_column="target"), max_iterations=2, call=boom)
print("iris (3-class) meta:", rep2.planner_meta, "| test acc:", round(rep2.attempts[0].qnn.test_accuracy, 2),
      "| baseline matched/full:", round(rep2.attempts[0].baseline.matched_accuracy, 2), round(rep2.attempts[0].baseline.full_accuracy, 2),
      "| attempts:", len(rep2.attempts))
print(rep2.attempts[0].evaluation.verdict)
