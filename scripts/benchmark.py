"""Hour-4 task for the Quantum dev: how long does one training run take vs qubits? Use it to set catalog.MAX_* limits."""
import sys, time, warnings
warnings.filterwarnings("ignore")
import pandas as pd
from autoqml.fallback import default_plan
from autoqml.profiler import profile_dataset
from autoqml.preprocess import prepare_data
from autoqml.quantum.trainer import train_qnn

df = pd.read_csv("data/breast_cancer.csv")
prof = profile_dataset(df, "target")
sampler = sys.argv[1] if len(sys.argv) > 1 else "reference"
print(f"sampler={sampler} | 100 train rows | COBYLA maxiter=30 | ZFeatureMap r2 + RealAmplitudes r2")
for n in (2, 4, 6, 8):
    p = default_plan(prof)
    p.n_qubits, p.train_subsample, p.optimizer.maxiter = n, 100, 30
    p.feature_map.reps, p.ansatz.name = 2, "RealAmplitudes"
    d = prepare_data(df, "target", p)
    t = time.time(); r = train_qnn(p, d, sampler)
    print(f"  n={n}: {time.time()-t:5.1f}s  ({(time.time()-t)/max(len(r.loss_history),1)*1000:.0f} ms/eval)  test_acc={r.test_accuracy:.2f}  cx={r.cx_count}", flush=True)
