"""Main hardware run: TFIM (n=6) depths x scales, GHZ (n=4,6) x scales, readout calibration.
One Sampler job, gate+measurement twirling, SHOTS per circuit. Repeat index from argv."""
import os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS_DIR = os.path.join(ROOT, "results", "hardware")
import json, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hw_common import *
from qiskit import ClassicalRegister
from qiskit_ibm_runtime import SamplerV2 as Sampler

REP = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SHOTS = 8000
BACKEND = "ibm_fez"
DEPTHS = [1, 2, 3, 4, 6, 8]
GHZ_N = [4, 6]
SCALES = [1, 3, 5]

svc = service()
backend = svc.backend(BACKEND)
chain6, cost6 = best_linear_chain(backend, 6)
print("chain6", chain6, round(cost6, 4))
pm = generate_preset_pass_manager(optimization_level=1, backend=backend, initial_layout=chain6)
pm0 = generate_preset_pass_manager(optimization_level=0, backend=backend)


def prepare(logical, n, basis_h=False):
    isa = pm.run(logical)
    phys = [int(p) for p in isa.layout.final_index_layout()[:n]]
    out = []
    for s in SCALES:
        f = fold(isa, s)
        f = pm0.run(f)
        if basis_h:
            for p in phys:
                f.h(p)  # X-basis readout for GHZ parity (noiseless ideal; small cost on device)
            f = pm0.run(f)
        cr = ClassicalRegister(n, "meas"); f.add_register(cr)
        f.barrier(phys)
        for i, p in enumerate(phys):
            f.measure(p, cr[i])
        out.append((f, dict(scale=s, ops={k: int(v) for k, v in f.count_ops().items()}, phys=phys)))
    return out


pubs, meta = [], []
for d in DEPTHS:
    for f, m in prepare(tfim_qc(6, d), 6):
        pubs.append(f); meta.append(dict(family="TFIM", n=6, depth=d, **m))
for nq in GHZ_N:
    # use first nq qubits of the chain
    pm_g = generate_preset_pass_manager(optimization_level=1, backend=backend, initial_layout=chain6[:nq])
    isa = pm_g.run(ghz_qc(nq))
    phys = [int(p) for p in isa.layout.final_index_layout()[:nq]]
    for s in SCALES:
        f = fold(isa, s); f = pm0.run(f)
        for p in phys: f.h(p)
        f = pm0.run(f)
        cr = ClassicalRegister(nq, "meas"); f.add_register(cr); f.barrier(phys)
        for i, p in enumerate(phys): f.measure(p, cr[i])
        pubs.append(f); meta.append(dict(family="GHZ", n=nq, depth=1, scale=s,
                                         ops={k: int(v) for k, v in f.count_ops().items()}, phys=phys))
# readout calibration: all-zero and all-one on the 6 chain qubits
for label in ("cal0", "cal1"):
    qc = QuantumCircuit(backend.num_qubits)
    if label == "cal1":
        for p in chain6: qc.x(p)
    qc = pm0.run(qc)
    cr = ClassicalRegister(6, "meas"); qc.add_register(cr); qc.barrier(chain6)
    for i, p in enumerate(chain6): qc.measure(p, cr[i])
    pubs.append(qc); meta.append(dict(family=label, n=6, depth=0, scale=0, ops={}, phys=chain6))

print("circuits:", len(pubs), "shots each:", SHOTS)
sampler = Sampler(mode=backend)
sampler.options.twirling.enable_gates = True
sampler.options.twirling.enable_measure = True
sampler.options.twirling.num_randomizations = 64
sampler.options.dynamical_decoupling.enable = False
job = sampler.run(pubs, shots=SHOTS)
print("job id", job.job_id())
json.dump(dict(job_id=job.job_id(), backend=BACKEND, rep=REP, shots=SHOTS, meta=meta, chain6=chain6),
          open(fos.path.join(RESULTS_DIR, "main_job_rep{REP}.json"), "w"), indent=1)
res = job.result()
out = []
for i, pub in enumerate(res):
    counts = pub.data.meas.get_counts()
    out.append(dict(**meta[i], counts=counts))
m = job.metrics()
json.dump(dict(job_id=job.job_id(), backend=BACKEND, rep=REP, shots=SHOTS, results=out, metrics=m,
               calibration_time=str(backend.properties().last_update_date)),
          open(fos.path.join(RESULTS_DIR, "main_result_rep{REP}.json"), "w"), indent=1, default=str)
for r in out:
    if r["family"] in ("TFIM", "GHZ"):
        n = r["n"]; z, tot = z_expectations(r["counts"], n)
        if r["family"] == "TFIM":
            print(r["family"], "d", r["depth"], "scale", r["scale"], "<Z3> = %.4f" % z[3])
        else:
            # parity from counts
            par = sum(c * (1 - 2 * (bin(int(bs.replace(' ', ''), 2)).count('1') % 2)) for bs, c in r["counts"].items()) / tot
            print(r["family"], "n", n, "scale", r["scale"], "<X..X> = %.4f" % par)
    else:
        z, tot = z_expectations(r["counts"], 6); print(r["family"], [round(x, 4) for x in z])
print("remaining:", svc.usage().get("usage_remaining_seconds"))
