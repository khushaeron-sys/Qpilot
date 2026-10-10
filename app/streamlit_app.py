"""Streamlit UI. Owner: Frontend/Pitch dev.  Run:  streamlit run app/streamlit_app.py"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


from autoqml.agent.orchestrator import run_autoqml
from autoqml.profiler import guess_target
from autoqml.quantum.builders import build_circuit
from autoqml.schemas import RunReport, RunRequest
import autoqml.fallback as fallback_mod

_ROOT = Path(__file__).resolve().parents[1]
CACHE = _ROOT / "data/cache/last_run.json"
DEMOS = {"Breast cancer (30 features, binary)": str(_ROOT / "data/breast_cancer.csv")}

st.set_page_config(page_title="AutoQML", page_icon="⚛️", layout="wide")
# --- Simple password gate ---
def check_password():
    def password_entered():
        if st.session_state["password"] == "qhack2026":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.markdown("""
    <style>
    .block-container { padding-top: 10vh; }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("""
        <div style="text-align:center; margin-bottom: 24px;">
            <div style="font-size: 48px;">⚛️</div>
            <div style="font-size: 26px; font-weight: 700; color: #F5F5F7; margin-top: 8px;">
                AutoQML
            </div>
            <div style="font-size: 14px; color: #9CA3AF; margin-top: 4px;">
                Natural Language → Quantum ML Pipeline
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#171B26; border:1px solid #262A35; border-radius:12px;
        padding:28px 24px; box-shadow: 0 8px 24px rgba(0,0,0,0.3);">
        """, unsafe_allow_html=True)

        st.text_input("Password", type="password", on_change=password_entered, key="password",
                       placeholder="Enter access password")

        if "password_correct" in st.session_state and not st.session_state["password_correct"]:
            st.markdown('<p style="color:#F87171; font-size:13px; margin-top:-8px;">Incorrect password, try again.</p>',
                        unsafe_allow_html=True)

        st.markdown("""
        <p style="color:#6B7280; font-size:12px; text-align:center; margin-top:16px;">
            Q-Hack India 2026 · IBM Quantum Track
        </p>
        </div>
        """, unsafe_allow_html=True)

    return False

if not check_password():
    st.stop()
# --- end password gate ---


PURPLE = "#A855F7"
BG = "#0B0E14"
CARD = "#171B26"
GREY = "#9CA3AF"

def style_fig(fig, ax):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#333842")

st.title("⚛️ AutoQML — natural language → quantum ML pipeline")
st.markdown(f"""
<div style="background:{CARD}; border-left:4px solid {PURPLE}; padding:16px 20px;
border-radius:6px; margin-bottom:20px; font-size:15px; line-height:1.5;">
You describe your ML task in plain English → an agent selects a NISQ-feasible
quantum architecture from a fixed Qiskit catalog → a real parameterized
quantum circuit trains on a simulator → we benchmark it against a classical
baseline on identical data.
</div>
""", unsafe_allow_html=True)

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
            st.warning(f"Demo file not found: '{DEMOS[demo]}'")

    target = prompt = None
    iters = 2
    if df is not None:
        st.header("2 · Task")
        guess = guess_target(df)
        target = st.selectbox("Target column", list(df.columns), index=list(df.columns).index(guess))
        prompt = st.text_area(
            "What should the model do?",
            value="Classify whether the tumor is malignant. Prioritize accuracy, keep it under 6 qubits.",
            height=100,
        )
        iters = st.slider("Max agent iterations", 1, 5, 2, help="How many planner retries before falling back")

    st.header("3 · Planner")
    use_fallback = st.checkbox("Skip LLM planner (use deterministic fallback)", value=True)

    run = st.button("🚀 Run AutoQML", use_container_width=True, type="primary")
    load_cached = False
    if CACHE.exists():
        load_cached = st.button("📺 Load last saved run (demo fallback)", use_container_width=True)

if load_cached and CACHE.exists():
    st.session_state["report"] = RunReport.model_validate_json(CACHE.read_text())

if run and df is not None:
    trace = []
    with st.status("AutoQML agent running…", expanded=True) as status:
        def on_event(kind, p):
            if kind == "profile":
                msg = f"📊 **Profiled** {p.n_rows} rows · {p.n_features_encoded} features · {p.n_classes} classes"
            elif kind == "plan":
                plan, meta = p
                msg = (f"🧠 **Planner ({meta['source']})**: {plan.n_qubits} qubits · {plan.dim_reduction} · "
                       f"{plan.feature_map.name}+{plan.ansatz.name} · {plan.optimizer.name}({plan.optimizer.maxiter})")
            elif kind == "training":
                msg = "⚛️ **Training** the parameterized quantum circuit on the simulator…"
            elif kind == "trained":
                msg = (f"✅ **Trained** · QNN test acc {p.qnn.test_accuracy:.1%} vs classical "
                       f"{p.baseline.matched_accuracy:.1%} · {p.qnn.train_seconds:.0f}s")
            else:
                msg = f"🔎 **Critic**: {p}"
            trace.append(msg)
            st.write(msg)

        try:
            report = run_autoqml(df, RunRequest(user_prompt=prompt, target_column=target),
                                  max_iterations=iters, on_event=on_event,
                                  use_fallback=use_fallback)
        except Exception as e:
            status.update(label="Cannot run", state="error")
            st.error(f"Pipeline failed: {e}")
            st.info("Try enabling 'Skip LLM planner' if this looks like an LLM/API issue.")
            st.stop()
        status.update(label="Done", state="complete", expanded=False)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(report.model_dump_json())
    st.session_state["report"], st.session_state["trace"] = report, trace

report = st.session_state.get("report")
if report is None:
    st.info("Upload a CSV (or pick a demo), describe the goal in plain English, and hit **Run AutoQML**.")
    st.stop()

final = report.attempts[-1]
q, b, ev, plan = final.qnn, final.baseline, final.evaluation, final.plan

st.success(f"Pipeline completed in {q.train_seconds:.1f}s — real {plan.n_qubits}-qubit "
           f"quantum circuit trained on simulator")

st.divider()

c1, c2, c3 = st.columns(3)
c1.metric("Quantum Accuracy", f"{q.test_accuracy:.1%}")
c2.metric("Classical (matched features)", f"{b.matched_accuracy:.1%}",
          delta=f"{(b.matched_accuracy - q.test_accuracy)*100:+.1f}pp")
c3.metric("Classical (all features)", f"{b.full_accuracy:.1%}")

st.divider()

left, right = st.columns(2)
with left:
    st.subheader("Quantum Circuit")
    try:
        circuit = build_circuit(plan)
        fig = circuit.draw(output="mpl", style="iqp")
        st.pyplot(fig, use_container_width=True)
    except Exception:
        st.code(str(plan.feature_map.name) + " + " + str(plan.ansatz.name))

with right:
    st.subheader("Chosen Architecture")
    st.markdown(f"""
- **Qubits:** {plan.n_qubits}
- **Feature Map:** {plan.feature_map.name} (reps={plan.feature_map.reps})
- **Ansatz:** {plan.ansatz.name} (reps={plan.ansatz.reps})
- **Optimizer:** {plan.optimizer.name} (maxiter={plan.optimizer.maxiter})
- **Dimensionality Reduction:** {plan.dim_reduction}

**Reasoning:** {plan.rationale}
""")

st.divider()

chart_left, chart_right = st.columns(2)
with chart_left:
    fig, ax = plt.subplots()
    ax.plot(q.loss_history, color=PURPLE, linewidth=2)
    ax.set_title("Training Loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    style_fig(fig, ax)
    st.pyplot(fig, use_container_width=True)

with chart_right:
    fig, ax = plt.subplots()
    labels = ["Quantum", "Classical\n(matched)", "Classical\n(full)"]
    values = [q.test_accuracy, b.matched_accuracy, b.full_accuracy]
    colors = [PURPLE, "#6B7280", "#9CA3AF"]
    ax.barh(labels, values, color=colors)
    ax.set_xlim(0, 1)
    ax.set_title("Quantum vs Classical")
    ax.set_xlabel("Accuracy")
    style_fig(fig, ax)
    st.pyplot(fig, use_container_width=True)

st.markdown(f"""
<hr style="margin-top:40px; opacity:0.2;">
<p style="text-align:center; color:{GREY}; font-size:12px;">
Q-Hack India 2026 · Built with Qiskit, Claude, and Streamlit
</p>
""", unsafe_allow_html=True)