"""Amplitude-damping robustness study (TFIM family, p2 = 0.01).

Noise after each k-qubit gate: k-qubit depolarizing (p_k) then single-qubit amplitude
damping with probability g_k = ETA * p_k on each of the gate's qubits.
  'twirled'  : amplitude damping replaced by its Pauli twirl -> total noise is Pauli; PEC exact.
  'untwirled': true amplitude damping; PEC built from the twirled model (model mismatch).
"""
import json, os, time
import numpy as np
from qsim import DensityMatrixSim, lam_from_p, gamma_from_lam, fold_global, gate_counts
from circuits import tfim_trotter, tfim_observable
from noise_general import *

ETA = 0.5
P2 = 0.01; P1 = P2 / 10
SCALES = [1, 3, 5]
OUT = "results"; os.makedirs(OUT, exist_ok=True)

# ---------- build per-gate-type Pauli models
def pauli_model(k):
    g = ETA * (P1 if k == 1 else P2)
    p = P1 if k == 1 else P2
    dep = depolarizing_probs(p, k)
    ad = tensor_probs([amp_damp_twirl_probs(g)] * k)
    total = compose_pauli(dep, ad, k)          # probability vector of composed Pauli channel
    inv, gamma = inverse_pauli_channel(total, k)
    return dict(total=total, inv=inv, gamma=gamma, dep=dep, g=g)

MODEL = {1: pauli_model(1), 2: pauli_model(2)}

# ---------- sanity checks of the Pauli machinery
_dep1 = depolarizing_probs(P1, 1)
_inv1, _g1 = inverse_pauli_channel(_dep1, 1)
assert abs(_g1 - gamma_from_lam(lam_from_p(P1, 1), 1)) < 1e-12
_inv2, _g2 = inverse_pauli_channel(depolarizing_probs(P2, 2), 2)
assert abs(_g2 - gamma_from_lam(lam_from_p(P2, 2), 2)) < 1e-12
# inverse composed with channel is identity
_id = compose_pauli(MODEL[2]["total"], MODEL[2]["inv"], 2)
assert np.allclose(_id, np.eye(16)[0])
print("Pauli machinery OK; gamma_1 =", MODEL[1]["gamma"], "gamma_2 =", MODEL[2]["gamma"])


def noise_twirled(g):
    k = len(g.qubits)
    return [(pauli_terms(MODEL[k]["total"], k), g.qubits)]


def noise_untwirled(g):
    k = len(g.qubits)
    out = [(pauli_terms(MODEL[k]["dep"], k), g.qubits)]
    for q in g.qubits:
        out.append((kraus_terms(amp_damp_kraus(MODEL[k]["g"])), (q,)))
    return out


def noise_untwirled_pec(g):
    k = len(g.qubits)
    return noise_untwirled(g) + [(pauli_terms(MODEL[k]["inv"], k), g.qubits)]


def noise_none(g):
    return []


records = []
n = 6
O, _, _ = tfim_observable(n)
sim = KrausDensityMatrixSim(n)
t0 = time.time()
for d in range(1, 13):
    circ = tfim_trotter(n, d)
    n1, n2 = gate_counts(circ)
    E0 = sim.expect(sim.run(circ, noise_none), O)
    gamma = MODEL[1]["gamma"] ** n1 * MODEL[2]["gamma"] ** n2
    rec = dict(depth=d, n1=n1, n2=n2, E0=E0, gamma=gamma,
               eps=n1 * (P1 + ETA * P1) + n2 * (P2 + 2 * ETA * P2))
    for s in SCALES:
        f = fold_global(circ, s)
        rec[f"tw_E{s}"] = sim.expect(sim.run(f, noise_twirled), O)
        rec[f"ad_E{s}"] = sim.expect(sim.run(f, noise_untwirled), O)
    rec["ad_pec_target"] = sim.expect(sim.run(circ, noise_untwirled_pec), O)
    records.append(rec)
    print(f"d={d} E0={E0:.4f} tw_E1={rec['tw_E1']:.4f} ad_E1={rec['ad_E1']:.4f} "
          f"PEC(ad)->{rec['ad_pec_target']:.4f} gamma^2={gamma**2:.3f} [{time.time()-t0:.0f}s]")

json.dump(dict(eta=ETA, p1=P1, p2=P2, model={k: dict(gamma=v["gamma"], g=v["g"]) for k, v in MODEL.items()},
               records=records), open(f"{OUT}/ampdamp.json", "w"), indent=1)
