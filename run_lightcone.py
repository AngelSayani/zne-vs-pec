"""Light-cone-restricted PEC: gate counts inside the backward light cone of a local
observable, epsilon_LC vs epsilon, gamma_LC, and exactness check on n=6/8."""
import json, numpy as np
from qsim import DensityMatrixSim, lam_from_p, gamma_from_lam, gate_counts
from circuits import tfim_trotter, tfim_observable

P2 = 0.01; P1 = P2 / 10


def light_cone_gates(circ, obs_qubits):
    """Return indices of gates in the backward light cone of obs_qubits (Heisenberg picture)."""
    cone = set(obs_qubits); idx = []
    for i in range(len(circ) - 1, -1, -1):
        g = circ[i]
        if cone & set(g.qubits):
            idx.append(i); cone |= set(g.qubits)
    return sorted(idx)


rows = []
for n in [6, 8, 12, 16, 24, 32, 48]:
    for d in [1, 2, 3, 4, 6, 8]:
        circ = tfim_trotter(n, d)
        lc = light_cone_gates(circ, {n // 2})
        n1, n2 = gate_counts(circ)
        n1c = sum(1 for i in lc if len(circ[i].qubits) == 1); n2c = sum(1 for i in lc if len(circ[i].qubits) == 2)
        eps, eps_lc = n1 * P1 + n2 * P2, n1c * P1 + n2c * P2
        g1, g2 = gamma_from_lam(lam_from_p(P1, 1), 1), gamma_from_lam(lam_from_p(P2, 2), 2)
        rows.append(dict(n=n, depth=d, n1=n1, n2=n2, n1_lc=n1c, n2_lc=n2c, eps=eps, eps_lc=eps_lc,
                         gamma2=(g1 ** n1 * g2 ** n2) ** 2, gamma2_lc=(g1 ** n1c * g2 ** n2c) ** 2))

# exactness check: noise only inside the cone gives identical <Z_mid> to noise everywhere (n=6,8; d=2,3)
checks = []
for n in [6, 8]:
    for d in [1, 2, 3]:
        circ = tfim_trotter(n, d); O = tfim_observable(n)[0]; lc = set(light_cone_gates(circ, {n // 2}))
        dm = DensityMatrixSim(n); l1, l2 = lam_from_p(P1, 1), lam_from_p(P2, 2)
        rho_all = dm.run(circ, l1, l2)
        rho = dm.zero_state()
        for i, g in enumerate(circ):
            rho = dm._apply_unitary(rho, g.mat, g.qubits)
            if i in lc:
                rho = dm._depolarize(rho, l1 if len(g.qubits) == 1 else l2, g.qubits)
        checks.append(dict(n=n, depth=d, E_all=dm.expect(rho_all, O), E_cone_only=dm.expect(rho, O),
                           frac_gates_in_cone=len(lc) / len(circ)))
        print(checks[-1])

VL, VQ = 5.0, 15.65625
for r in rows:
    r["pec_dominates_lin_full"] = r["gamma2"] <= VL; r["pec_dominates_lin_lc"] = r["gamma2_lc"] <= VL
    r["pec_dominates_quad_full"] = r["gamma2"] <= VQ; r["pec_dominates_quad_lc"] = r["gamma2_lc"] <= VQ
json.dump(dict(rows=rows, checks=checks), open("results/lightcone.json", "w"), indent=1)
for r in rows:
    if r["n"] in (6, 16, 48):
        print(f"n={r['n']:2d} d={r['depth']} eps={r['eps']:.2f} eps_lc={r['eps_lc']:.2f} gamma2={r['gamma2']:.3g} gamma2_lc={r['gamma2_lc']:.3g}")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5, "figure.dpi": 150, "savefig.bbox": "tight"})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
ax = axes[0]
for n, mk in zip([6, 12, 24, 48], ["o", "s", "^", "D"]):
    sel = [r for r in rows if r["n"] == n]
    ax.semilogy([r["depth"] for r in sel], [r["gamma2"] for r in sel], mk + "--", ms=3.5, mfc="none", color="#d62728", lw=0.8, label=rf"$\gamma^2$, $n={n}$")
    ax.semilogy([r["depth"] for r in sel], [r["gamma2_lc"] for r in sel], mk + "-", ms=3.5, color="#1f77b4", lw=0.8, label=rf"$\gamma_{{\rm LC}}^2$, $n={n}$" if n == 6 else None)
ax.axhline(VL, color="0.3", lw=0.8); ax.axhline(VQ, color="0.3", lw=0.8)
ax.text(1.05, VL * 1.15, r"$V_{\rm lin}$", fontsize=7); ax.text(1.05, VQ * 1.15, r"$V_{\rm quad}$", fontsize=7)
ax.set_xlabel("Trotter steps"); ax.set_ylabel("PEC overhead"); ax.set_title(r"full-circuit vs light-cone PEC, $p_2=0.01$", fontsize=9)
ax.legend(frameon=False, fontsize=6.5, ncol=2); ax.set_ylim(1, 1e4)
ax = axes[1]
ns = sorted(set(r["n"] for r in rows)); ds = sorted(set(r["depth"] for r in rows))
Z = np.array([[next(r for r in rows if r["n"] == n and r["depth"] == d)["eps_lc"] / next(r for r in rows if r["n"] == n and r["depth"] == d)["eps"] for d in ds] for n in ns])
im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(len(ds))); ax.set_xticklabels(ds); ax.set_yticks(range(len(ns))); ax.set_yticklabels(ns)
ax.set_xlabel("Trotter steps"); ax.set_ylabel("qubits $n$"); ax.set_title(r"$\epsilon_{\rm LC}/\epsilon$ for $Z_{n/2}$", fontsize=9)
plt.colorbar(im, ax=ax, fraction=0.046)
fig.savefig("figures/fig7_lightcone.pdf"); plt.close(fig)
