"""Streamlit UI. Owner: Frontend/Pitch dev.  Run:  streamlit run app/streamlit_app.py"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from autoqml.agent.orchestrator import run_autoqml
from autoqml.profiler import guess_target
from autoqml.quantum.builders import build_circuit
from autoqml.schemas import RunReport, RunRequest
import autoqml.fallback as fallback_mod

# Anchor all data paths to the project root (parent of the app/ directory)
_ROOT = Path(__file__).resolve().parents[1]
CACHE = _ROOT / "data/cache/last_run.json"
DEMOS = {"Breast cancer (30 features, binary)": str(_ROOT / "data/breast_cancer.csv")}

st.set_page_config(page_title="AutoQML", page_icon="⚛️", layout="wide")
st.title("⚛️ AutoQML — natural language → quantum ML pipeline")

# 1. Styled info banner
st.markdown(
    """
    <div style="
        background:#171B26;
        border-left:4px solid #A855F7;
        padding:16px 20px;
        border-radius:6px;
        margin-bottom:20px;
        color:#E2E8F0;
        font-size:0.95rem;
        line-height:1.6;
    ">
        You describe your ML task in plain English &rarr; an agent selects a NISQ-feasible quantum
        architecture from a fixed Qiskit catalog &rarr; a real parameterized quantum circuit trains
        on a simulator &rarr; we benchmark it against a classical baseline on identical data.
    </div>
    """,
    unsafe_allow_html=True,
)

# ───────────── sidebar: inputs ─────────────
with st.sidebar:
    st.header("1 · Data")
    up = st.file_uploader("Upload a CSV", type="csv")
    demo = st.selectbox("…or pick a demo dataset", ["(none)"] + list(DEMOS))
    df = None
    if up:
        df = pd.read_csv(up)
    elif demo != "(none)":
        try:
            df = pd.read_csv(DEMOS[demo])
        except FileNotFoundError:
            st.warning(f"Demo file not found: `{DEMOS[demo]}`")

    target = prompt = None
    iters = 2
    if df is not None:
        st.header("2 · Task")
        guess = guess_target(df)
        target = st.selectbox("Target column", list(df.columns), index=list(df.columns).index(guess))
        prompt = st.text_area("What should the model do?", height=120,
                              value="Classify whether the tumor is malignant. Prioritize accuracy, keep it under 6 qubits.")
        iters = st.slider("Max agent iterations", 1, 3, 2, help="1 = plan once. 2+ lets the critic revise the plan.")

    st.header("3 · Planner")
    use_fallback = st.checkbox(
        "Skip LLM planner (use deterministic fallback)",
        value=False,
        help="Bypass the Claude API and use a fixed, rule-based quantum plan. Useful when no API key is set or for reproducible runs.",
    )
    if use_fallback:
        st.info("⚙️ Running in fallback mode.", icon="⚙️")

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
        # 3. Broader exception handler
        try:
            report = run_autoqml(df, RunRequest(user_prompt=prompt, target_column=target),
                                 max_iterations=iters, on_event=on_event,
                                 use_fallback=use_fallback)
        except Exception as e:
            status.update(label="Pipeline failed", state="error")
            st.error(f"Pipeline failed: {e}")
            st.info("Try enabling the fallback planner if this is an LLM API issue.")
            st.stop()
        status.update(label="Done", state="complete", expanded=False)
    CACHE.parent.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
    CACHE.write_text(report.model_dump_json())
    st.session_state["report"], st.session_state["trace"] = report, trace

# ───────────── results ─────────────
report = st.session_state.get("report")
if report is None:
    st.info("Upload a CSV (or pick a demo), describe the goal in plain English, and hit **Run AutoQML**.")
    st.stop()

final = report.attempts[-1]
q, b, ev, plan = final.qnn, final.baseline, final.evaluation, final.plan

# 2a: completion banner
st.success(
    f"Pipeline completed in {q.train_seconds:.1f}s — "
    f"real {plan.n_qubits}-qubit quantum circuit trained on simulator"
)

# 2b
st.divider()

# 2c: 3-column metrics
m1, m2, m3 = st.columns(3)
m1.metric("Quantum Accuracy", f"{q.test_accuracy:.1%}")
m2.metric("Classical (matched features)", f"{b.matched_accuracy:.1%}")
m3.metric("Classical (all features)", f"{b.full_accuracy:.1%}")

# 2d
st.divider()

# 2e: circuit + architecture side-by-side
col_circ, col_arch = st.columns(2)

with col_circ:
    st.subheader("Quantum Circuit")
    try:
        qc = build_circuit(plan)
        fig_circ = qc.draw(output="mpl", style="iqp")
        st.pyplot(fig_circ)
        plt.close(fig_circ)
    except Exception:
        try:
            qc = build_circuit(plan)
            st.code(str(qc))
        except Exception as circ_err:
            st.warning(f"Could not render circuit: {circ_err}")

with col_arch:
    st.subheader("Chosen Architecture")
    st.markdown(
        f"""
- **Qubits:** {plan.n_qubits}
- **Feature Map:** {plan.feature_map.name} (reps={plan.feature_map.reps})
- **Ansatz:** {plan.ansatz.name} (reps={plan.ansatz.reps})
- **Optimizer:** {plan.optimizer.name} (maxiter={plan.optimizer.maxiter})
- **Reasoning:** {plan.rationale}
        """
    )

# 2f
st.divider()

# 2g: dark matplotlib charts
def _dark_fig():
    """Return a (fig, ax) pair pre-styled with a dark background."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor("#0B0E14")
    ax.set_facecolor("#0B0E14")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2D3748")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    return fig, ax


chart_l, chart_r = st.columns(2)

with chart_l:
    fig_loss, ax_loss = _dark_fig()
    ax_loss.plot(q.loss_history, color="#A855F7", linewidth=2)
    ax_loss.set_title("Training Loss", fontsize=13, fontweight="bold")
    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel("Loss")
    st.pyplot(fig_loss)
    plt.close(fig_loss)

with chart_r:
    fig_acc, ax_acc = _dark_fig()
    labels = [
        "Quantum VQC",
        f"Classical\n({b.matched_n_features} feat.)",
        f"Classical\n(all {b.full_n_features} feat.)",
    ]
    values = [q.test_accuracy, b.matched_accuracy, b.full_accuracy]
    bars = ax_acc.barh(labels, values, color=["#A855F7", "#6B7280", "#4B5563"], height=0.5)
    ax_acc.set_xlim(0, 1)
    ax_acc.set_xlabel("Accuracy")
    ax_acc.set_title("Accuracy Comparison", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, values):
        ax_acc.text(
            min(val + 0.01, 0.97), bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}", va="center", color="white", fontsize=10,
        )
    st.pyplot(fig_acc)
    plt.close(fig_acc)

# ── detailed tabs ──
t_res, t_circ, t_trace, t_plan = st.tabs(["📈 Detailed Results", "🧬 Circuit (decomposed)", "🤖 Agent trace", "🧾 Plan (JSON)"])

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
    st.caption("Feature map (data encoding, input angles x[i]) followed by the trainable ansatz θ[i].")
    try:
        qc2 = build_circuit(plan)
        st.pyplot(qc2.decompose(reps=2).draw("mpl", fold=28, scale=0.7))
    except Exception:
        try:
            qc2 = build_circuit(plan)
            st.code(str(qc2.decompose(reps=2).draw("text", fold=120)))
        except Exception as e2:
            st.warning(f"Circuit render failed: {e2}")

with t_trace:
    for m in st.session_state.get("trace", ["(trace is only kept for live runs)"]):
        st.markdown(m)
    st.json(report.planner_meta)

with t_plan:
    st.json(plan.model_dump())

# 4. Footer
st.markdown(
    """
    <hr style="border:none;border-top:1px solid #2D3748;margin:40px 0 12px 0;">
    <p style="text-align:center;color:#6B7280;font-size:12px;margin:0;">
        Q-Hack India 2026 &middot; Built with Qiskit, Claude, and Streamlit
    </p>
    """,
    unsafe_allow_html=True,
)
