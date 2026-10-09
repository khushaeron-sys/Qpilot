# AutoQML: Natural Language -> Quantum ML pipeline

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...        # without a key the agent falls back to a deterministic plan
python scripts/make_demo_data.py            # writes data/breast_cancer.csv, data/iris.csv
PYTHONPATH=. python scripts/benchmark.py    # runtime vs qubits on YOUR laptop -> set catalog.MAX_* limits
PYTHONPATH=. python tests/test_agent_flow.py   # agent loop with a scripted fake Claude (no API key needed)
streamlit run app/streamlit_app.py
```

Layout: `autoqml/catalog.py` (what the LLM may choose) · `schemas.py` (contracts) · `agent/` (WS1) · `quantum/` (WS2) ·
`profiler|preprocess|baseline|evaluator.py` (WS3) · `app/` (WS4).
