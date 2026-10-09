"""Streamlit UI. Owner: Frontend/Pitch dev.  Run:  streamlit run app/streamlit_app.py"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import pandas as pd
import streamlit as st

from autoqml.agent.orchestrator import run_autoqml
from autoqml.profiler import guess_target
from autoqml.quantum.builders import build_circuit
from autoqml.schemas import RunReport, RunRequest

CACHE = Path("data/cache/last_run.json")
DEMOS = {"Breast cancer (30 features, binary)": "data/breast_cancer.csv"}

st.set_page_config(page_title="AutoQML", page_icon="⚛️", layout="wide")
st.title("⚛️ AutoQML — natural language → quantum ML pipeline")
st.caption("Claude plans the circuit from a fixed Qiskit catalog · Qiskit trains a real parameterized quantum circuit · "
           "a classical model keeps it honest.")

# ───────────── sidebar: inputs ─────────────
with st.sidebar:
    st.header("1 · Data")
    up = st.file_uploader("Upload a CSV", type="csv")
    demo = st.selectbox("…or pick a demo dataset", ["(none)"] + list(DEMOS))
    df = pd.read_csv(up) if up else (pd.read_csv(DEMOS[demo]) if demo != "(none)" else None)

    target = prompt = None
    if df is not None:
        st.header("2 · Task")
        guess = guess_target(df)
        target = st.selectbox("Target column", list(df.columns), index=list(df.columns).index(guess))
        prompt = st.text_area("What should the model do?", height=120,
                              value="Classify whether the tumor is malignant. Prioritize accuracy, keep it under 6 qubits.")
        iters = st.slider("Max agent iterations", 1, 3, 2, help="1 = plan once. 2+ lets the critic revise the plan.")
    run = st.button("🚀 Run AutoQML", type="primary", disabled=df is None, use_container_width=True)
    if CACHE.exists() and st.button("📼 Load last saved run (demo fallback)", use_container_width=True):
        st.session_state["report"] = RunReport.model_validate_json(CACHE.read_text())

# ───────────── run the agent ─────────────
if run and df is not None:
    trace = []
    with st.status("AutoQML agent running…", expanded=True) as status:
        def on_event(kind, p):
            if kind == "profile":
                msg = f"📊 **Profiled** {p.n_rows} rows · {p.n_features_encoded} features · {p.n_classes} classes"
            elif kind == "plan":
                plan, meta = p
                msg = (f"🧠 **Planner ({meta['source']})**: {plan.n_qubits} qubits · {plan.dim_reduction} · "
                       f"{plan.feature_map.name} + {plan.ansatz.name} · {plan.optimizer.name}({plan.optimizer.maxiter})")
            elif kind == "training":
                msg = "⚛️ **Training** the parameterized quantum circuit on the simulator…"
            elif kind == "trained":
                msg = (f"✅ **Trained** · QNN test acc {p.qnn.test_accuracy:.1%} vs classical "
                       f"{p.baseline.matched_accuracy:.1%} · {p.qnn.train_seconds:.0f}s")
            else:
                msg = f"🔎 **Critic**: {p}"
            trace.append(msg); st.write(msg)
        try:
            report = run_autoqml(df, RunRequest(user_prompt=prompt, target_column=target),
                                 max_iterations=iters, on_event=on_event)
        except ValueError as e:
            status.update(label="Cannot run", state="error"); st.error(str(e)); st.stop()
        status.update(label="Done", state="complete", expanded=False)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(report.model_dump_json())
    st.session_state["report"], st.session_state["trace"] = report, trace

# ───────────── results ─────────────
report = st.session_state.get("report")
if report is None:
    st.info("Upload a CSV (or pick a demo), describe the goal in plain English, and hit **Run AutoQML**.")
    st.stop()

final = report.attempts[-1]
q, b, ev, plan = final.qnn, final.baseline, final.evaluation, final.plan

c1, c2, c3, c4 = st.columns(4)
c1.metric("⚛️ Quantum VQC (test acc)", f"{q.test_accuracy:.1%}")
c2.metric(f"Classical · same {b.matched_n_features} features", f"{b.matched_accuracy:.1%}",
          delta=f"{ev.delta_vs_matched*100:+.1f} pts (QNN − classical)", delta_color="off")
c3.metric(f"Classical · all {b.full_n_features} features", f"{b.full_accuracy:.1%}")
c4.metric("Circuit", f"{q.n_qubits} qubits · {q.n_params} params", f"depth {q.circuit_depth} · {q.cx_count} CX", delta_color="off")
st.info(f"**Verdict:** {ev.verdict}  \n**Agent rationale:** {plan.rationale}")

t_res, t_circ, t_trace, t_plan = st.tabs(["📈 Results", "🧬 Circuit", "🤖 Agent trace", "🧾 Plan (JSON)"])

with t_res:
    a, bcol = st.columns(2)
    with a:
        st.subheader("Training loss (per objective evaluation)")
        st.line_chart(pd.DataFrame({"cross-entropy loss": q.loss_history}))
        st.caption(f"Loss trend: **{ev.loss_trend}** · train/test gap: {ev.overfit_gap*100:+.1f} pts · trained in {q.train_seconds:.0f}s")
    with bcol:
        st.subheader("Test accuracy")
        st.bar_chart(pd.DataFrame({"accuracy": [q.test_accuracy, b.matched_accuracy, b.full_accuracy]},
                                  index=["Quantum VQC", f"Classical ({b.matched_n_features} feat.)", f"Classical ({b.full_n_features} feat.)"]))
        st.caption("Test set ≤100 rows → ±4 pt noise. Treat small gaps as ties.")
    if len(report.attempts) > 1:
        st.subheader("Agent iterations")
        st.dataframe(pd.DataFrame([{
            "attempt": i + 1, "feature_map": f"{x.plan.feature_map.name} r{x.plan.feature_map.reps}",
            "ansatz": f"{x.plan.ansatz.name} r{x.plan.ansatz.reps}", "optimizer": f"{x.plan.optimizer.name}({x.plan.optimizer.maxiter})",
            "test_acc": round(x.qnn.test_accuracy, 3), "critic": x.critic_note or ""} for i, x in enumerate(report.attempts)]),
            use_container_width=True, hide_index=True)

with t_circ:
    qc = build_circuit(plan)
    st.caption("Feature map (data encoding, input angles x[i]) followed by the trainable ansatz θ[i].")
    try:
        st.pyplot(qc.decompose(reps=2).draw("mpl", fold=28, scale=0.7))
    except Exception:
        st.code(str(qc.decompose(reps=2).draw("text", fold=120)))

with t_trace:
    for m in st.session_state.get("trace", ["(trace is only kept for live runs)"]):
        st.markdown(m)
    st.json(report.planner_meta)

with t_plan:
    st.json(plan.model_dump())
