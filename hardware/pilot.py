"""Pilot: TFIM n=6, d=1, scales 1,3,5, twirled, 2000 shots each. Measures cost per circuit."""
import os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS_DIR = os.path.join(ROOT, "results", "hardware")
import json, time, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hw_common import *
from qiskit_ibm_runtime import SamplerV2 as Sampler

svc = service()
backend = svc.backend("ibm_fez")
n = 6
chain, cost = best_linear_chain(backend, n)
print("chain", chain, "summed err", round(cost, 4))
pm = generate_preset_pass_manager(optimization_level=1, backend=backend, initial_layout=chain)
pm0 = generate_preset_pass_manager(optimization_level=0, backend=backend)

logical = tfim_qc(n, 1)
isa = pm.run(logical)
circs = []
meta = []
for s in (1, 3, 5):
    f = fold(isa, s)
    f = pm0.run(f)  # decompose inverses into native gates, no cancellation
    phys = isa.layout.final_index_layout()[:n]  # physical qubit of each logical qubit
    from qiskit import ClassicalRegister
    cr = ClassicalRegister(n, "meas"); f.add_register(cr)
    f.barrier(phys)
    for i, p in enumerate(phys):
        f.measure(p, cr[i])
    ops = dict(f.count_ops())
    meta.append(dict(scale=s, ops=ops, depth=f.depth()))
    circs.append(f)
    print("scale", s, ops)

sampler = Sampler(mode=backend)
sampler.options.twirling.enable_gates = True
sampler.options.twirling.enable_measure = True
sampler.options.twirling.num_randomizations = 32
sampler.options.dynamical_decoupling.enable = False
job = sampler.run(circs, shots=2000)
print("job id", job.job_id())
json.dump(dict(job_id=job.job_id(), backend="ibm_fez", chain=chain, phys=list(map(int,phys)), meta=meta, logical_depth=1),
          open(os.path.join(RESULTS_DIR, "pilot_job.json"), "w"), indent=1)
res = job.result()
out = []
for i, pub in enumerate(res):
    counts = pub.data.meas.get_counts()
    z, tot = z_expectations(counts, n)
    out.append(dict(scale=meta[i]["scale"], z=list(z), shots=tot))
    print("scale", meta[i]["scale"], "<Z_3> =", round(z[3], 4), "shots", tot)
m = job.metrics()
print("usage seconds:", m.get("usage", {}).get("quantum_seconds"), "| metrics keys:", list(m.keys()))
json.dump(dict(results=out, metrics=m), open(os.path.join(RESULTS_DIR, "pilot_result.json"), "w"), indent=1, default=str)
print("remaining:", svc.usage().get("usage_remaining_seconds"))
