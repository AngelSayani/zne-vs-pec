"""Compute exact noisy expectation values for all circuit configurations.

Output: results/exact.json with, for every (family, size, depth, p2):
  E0 (ideal), E_lambda for lambda in {1,3,5}, gate counts, gamma for exact
  calibration, and PEC targets / gammas for miscalibration deltas.
"""
import json, os, time, itertools
import numpy as np
from qsim import DensityMatrixSim, lam_from_p, gamma_from_lam, fold_global, gate_counts
from circuits import *

OUT = "results"
os.makedirs(OUT, exist_ok=True)

P2_GRID = [0.002, 0.005, 0.01, 0.015, 0.02, 0.03]
RATIO = 10.0  # p2 / p1
DELTAS = [-0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3]
SCALES = [1, 3, 5]

configs = []
for d in range(1, 13):
    configs.append(("TFIM", 6, d))
for L in range(1, 9):
    configs.append(("HEA", 6, L))
for n in range(2, 9):
    configs.append(("GHZ", n, 1))


def build(fam, n, d):
    if fam == "TFIM":
        return tfim_trotter(n, d), tfim_observable(n)[0]
    if fam == "HEA":
        return hea(n, d), hea_observable(n)[0]
    if fam == "GHZ":
        return ghz(n), ghz_observable(n)[0]


records = []
t0 = time.time()
for fam, n, d in configs:
    circ, O = build(fam, n, d)
    dm = DensityMatrixSim(n)
    n1, n2 = gate_counts(circ)
    E0 = dm.expect(dm.run(circ, 1.0, 1.0), O)
    folded = {s: fold_global(circ, s) for s in SCALES}
    for p2 in P2_GRID:
        p1 = p2 / RATIO
        l1, l2 = lam_from_p(p1, 1), lam_from_p(p2, 2)
        E = {s: dm.expect(dm.run(folded[s], l1, l2), O) for s in SCALES}
        gamma = gamma_from_lam(l1, 1) ** n1 * gamma_from_lam(l2, 2) ** n2
        eps = n1 * p1 + n2 * p2
        mis = {}
        for delta in DELTAS:
            l1h, l2h = lam_from_p(p1 * (1 + delta), 1), lam_from_p(p2 * (1 + delta), 2)
            Eres = dm.expect(dm.run(circ, l1 / l1h, l2 / l2h), O)
            gh = gamma_from_lam(l1h, 1) ** n1 * gamma_from_lam(l2h, 2) ** n2
            mis[str(delta)] = {"E_target": Eres, "gamma_hat": gh}
        records.append(dict(family=fam, n=n, depth=d, p1=p1, p2=p2, n1=n1, n2=n2,
                            E0=E0, E1=E[1], E3=E[3], E5=E[5], gamma=gamma, eps=eps,
                            miscal=mis))
    print(f"{fam} n={n} d={d} gates=({n1},{n2}) E0={E0:.4f}  [{time.time()-t0:.1f}s]")

with open(f"{OUT}/exact.json", "w") as f:
    json.dump(records, f, indent=1)
print("records:", len(records))
