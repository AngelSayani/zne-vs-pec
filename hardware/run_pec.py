"""PEC on hardware via Runtime Estimator with learned Pauli-Lindblad noise (TFIM d=2,4 and GHZ n=6)."""
import os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS_DIR = os.path.join(ROOT, "results", "hardware")
import json, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hw_common import *
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import EstimatorV2 as Estimator

svc = service()
backend = svc.backend("ibm_fez")
chain6, _ = best_linear_chain(backend, 6)
pm = generate_preset_pass_manager(optimization_level=1, backend=backend, initial_layout=chain6)

pubs, meta = [], []
for d in (2, 4):
    isa = pm.run(tfim_qc(6, d))
    phys = [int(p) for p in isa.layout.final_index_layout()[:6]]
    obs = SparsePauliOp.from_sparse_list([("Z", [phys[3]], 1.0)], num_qubits=backend.num_qubits)
    pubs.append((isa, obs)); meta.append(dict(family="TFIM", n=6, depth=d, phys=phys,
                                            ops={k: int(v) for k, v in isa.count_ops().items()}))
isa = pm.run(ghz_qc(6))
phys = [int(p) for p in isa.layout.final_index_layout()[:6]]
obs = SparsePauliOp.from_sparse_list([("XXXXXX", phys, 1.0)], num_qubits=backend.num_qubits)
pubs.append((isa, obs)); meta.append(dict(family="GHZ", n=6, depth=1, phys=phys,
                                        ops={k: int(v) for k, v in isa.count_ops().items()}))

results = {}
for mode in ("raw", "pec"):
    est = Estimator(mode=backend)
    est.options.resilience_level = 0
    est.options.twirling.enable_gates = True
    est.options.twirling.enable_measure = True
    est.options.resilience.measure_mitigation = True   # TREX readout mitigation in both modes
    if mode == "pec":
        est.options.resilience.pec_mitigation = True
        est.options.resilience.pec.max_overhead = 100
        est.options.resilience.layer_noise_learning.num_randomizations = 32
        est.options.resilience.layer_noise_learning.shots_per_randomization = 128
        est.options.resilience.layer_noise_learning.layer_pair_depths = [0, 1, 2, 4, 16, 32]
    job = est.run(pubs, precision=0.01)
    print(mode, "job id", job.job_id())
    res = job.result()
    out = []
    for i, pub in enumerate(res):
        ev = float(pub.data.evs); std = float(pub.data.stds)
        md = pub.metadata
        out.append(dict(**meta[i], ev=ev, std=std, metadata=md))
        print(mode, meta[i]["family"], "d", meta[i]["depth"], "ev = %.4f +- %.4f" % (ev, std),
              "| shots:", md.get("shots"), "| overhead:", (md.get("resilience") or {}).get("sampling_overhead"))
    results[mode] = dict(job_id=job.job_id(), results=out, metrics=job.metrics(), result_metadata=res.metadata)
    print("remaining:", svc.usage().get("usage_remaining_seconds"))
json.dump(results, open(os.path.join(RESULTS_DIR, "pec_result.json"), "w"), indent=1, default=str)
