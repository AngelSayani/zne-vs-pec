"""Analyse hardware runs: readout correction, ZNE estimators + bootstrap variance,
depolarizing-model fit, comparison with exact predictions, PEC results."""
import os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS_DIR = os.path.join(ROOT, "results", "hardware")
import json, sys, glob
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from estimators import *
from qsim import DensityMatrixSim, lam_from_p, fold_global, gate_counts
from circuits import tfim_trotter, tfim_observable, ghz, ghz_observable
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5, "figure.dpi": 150, "savefig.bbox": "tight"})
COL = {"raw": "0.5", "ZNE-lin": "#1f77b4", "ZNE-quad": "#ff7f0e", "ZNE-exp0": "#2ca02c", "ZNE-exp3": "#9467bd", "PEC": "#d62728"}
rng = np.random.default_rng(7)

reps = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "main_result_rep*.json")))]
pec = json.load(open(os.path.join(RESULTS_DIR, "pec_result.json")))
R = len(reps); print("repeats:", R, "shots/circuit:", reps[0]["shots"])


def counts_to_vals(counts, n, kind):
    """Return per-shot observable values array (+-1) expanded from counts (for bootstrap)."""
    vals = []
    for bs, c in counts.items():
        bits = bs.replace(" ", "")[::-1]
        if kind == "Z3":
            v = 1 - 2 * int(bits[3])
        else:  # parity of all n
            v = 1 - 2 * (sum(int(b) for b in bits[:n]) % 2)
        vals.append((v, c))
    return vals


def mean_from_vals(vals):
    tot = sum(c for _, c in vals); return sum(v * c for v, c in vals) / tot


def readout_factor(rep, qubits_idx):
    """Symmetric flip probabilities from cal0/cal1; return prod(1-2q) over observable qubits."""
    c0 = [r for r in rep["results"] if r["family"] == "cal0"][0]["counts"]
    c1 = [r for r in rep["results"] if r["family"] == "cal1"][0]["counts"]
    def zexp(counts):
        tot = sum(counts.values()); z = np.zeros(6)
        for bs, c in counts.items():
            bits = bs.replace(" ", "")[::-1]
            for i in range(6): z[i] += c * (1 - 2 * int(bits[i]))
        return z / tot
    z0, z1 = zexp(c0), zexp(c1)
    q = (1 - z0 + 1 + z1) / 4  # average flip prob (0->1 and 1->0)
    return float(np.prod([1 - 2 * q[i] for i in qubits_idx])), q


# exact ideal values and simulated depolarizing predictions
dm6 = DensityMatrixSim(6)
ideal = {}
for d in [1, 2, 3, 4, 6, 8]:
    c = tfim_trotter(6, d); O = tfim_observable(6)[0]
    ideal[("TFIM", 6, d)] = (dm6.expect(dm6.run(c, 1, 1), O), c, O)
for nq in [4, 6]:
    dm = DensityMatrixSim(nq); c = ghz(nq); O = ghz_observable(nq)[0]
    ideal[("GHZ", nq, 1)] = (dm.expect(dm.run(c, 1, 1), O), c, O)

rows = []
for rep_i, rep in enumerate(reps):
    for key in ideal:
        fam, nq, d = key
        obs_q = [3] if fam == "TFIM" else list(range(nq))
        f_ro, qflip = readout_factor(rep, obs_q)
        E = {}; V = {}
        for s in (1, 3, 5):
            r = [x for x in rep["results"] if x["family"] == fam and x["n"] == nq and x["depth"] == d and x["scale"] == s][0]
            vals = counts_to_vals(r["counts"], nq, "Z3" if fam == "TFIM" else "par")
            E[s] = mean_from_vals(vals) / f_ro
            V[s] = vals
        E0 = ideal[key][0]
        row = dict(rep=rep_i, family=fam, n=nq, depth=d, E0=E0, E1=E[1], E3=E[3], E5=E[5], ro_factor=f_ro)
        # ZNE estimators + bootstrap over shots (B=400) using the actual 8000 shots per scale
        for name, (sc, est) in ZNE_METHODS.items():
            Ev = [E[s] for s in sc]; row[name] = est(Ev)
            boot = []
            for _ in range(400):
                Eb = []
                for s in sc:
                    vals = V[s]; tot = sum(c for _, c in vals)
                    p = np.array([c for _, c in vals]) / tot; v = np.array([v for v, _ in vals])
                    draw = rng.multinomial(tot, p); Eb.append((draw * v).sum() / tot / f_ro)
                boot.append(est(Eb))
            boot = np.array(boot); boot = boot[np.isfinite(boot)]
            row[name + "_bootstd"] = float(boot.std())
            # predicted std from delta method with N_total = 3*8000 or 2*8000
            Ntot = len(sc) * 8000
            var, _ = delta_var(Ev, sc, est, Ntot); row[name + "_predstd"] = float(np.sqrt(var)) / f_ro
        row["raw"] = E[1]; row["raw_bootstd"] = float(np.sqrt((1 - E[1] ** 2) / 8000)) / f_ro
        row["kappa"] = float(-0.5 * np.log(abs(E[3] / E[1])))
        rows.append(row)

# depolarizing fit: find p2 (p1=p2/10) s.t. simulated E1 matches hardware E1 (per circuit, rep-averaged)
fit = {}
for key in ideal:
    fam, nq, d = key
    E1_hw = np.mean([r["E1"] for r in rows if (r["family"], r["n"], r["depth"]) == key])
    c, O = ideal[key][1], ideal[key][2]; dm = DensityMatrixSim(nq)
    lo, hi = 1e-5, 0.1
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        e = dm.expect(dm.run(c, lam_from_p(mid / 10, 1), lam_from_p(mid, 2)), O)
        if e > E1_hw: lo = mid
        else: hi = mid
    p2 = np.sqrt(lo * hi)
    pred = {s: dm.expect(dm.run(fold_global(c, s), lam_from_p(p2 / 10, 1), lam_from_p(p2, 2)), O) for s in (1, 3, 5)}
    n1, n2 = gate_counts(c)
    fit[key] = dict(p2_fit=float(p2), pred=pred, E3_hw=np.mean([r["E3"] for r in rows if (r["family"], r["n"], r["depth"]) == key]),
                    E5_hw=np.mean([r["E5"] for r in rows if (r["family"], r["n"], r["depth"]) == key]), n1=n1, n2=n2)
    print(key, "p2_fit=%.4f" % p2, "pred E3,E5 = %.3f, %.3f" % (pred[3], pred[5]), "hw E3,E5 = %.3f, %.3f" % (fit[key]["E3_hw"], fit[key]["E5_hw"]))

# summary table (rep-averaged, with rep spread)
import collections
summary = []
for key in ideal:
    sel = [r for r in rows if (r["family"], r["n"], r["depth"]) == key]
    s = dict(family=key[0], n=key[1], depth=key[2], E0=sel[0]["E0"])
    for m in ["raw", "ZNE-lin", "ZNE-quad", "ZNE-exp0", "ZNE-exp3"]:
        vals = np.array([r[m] for r in sel])
        s[m] = float(vals.mean()); s[m + "_repstd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        s[m + "_bootstd"] = float(np.mean([r[m + "_bootstd"] for r in sel]))
        if m != "raw": s[m + "_predstd"] = float(np.mean([r[m + "_predstd"] for r in sel]))
        s[m + "_bias"] = s[m] - s["E0"]
        # empirical MSE across repeats at the actual budget (24000 shots for 3-scale, 16000 for 2-scale)
        s[m + "_mse_emp"] = float(np.mean((vals - s["E0"]) ** 2))
    s["kappa"] = float(np.mean([r["kappa"] for r in sel]))
    summary.append(s)
    print(f"{key}: E0={s['E0']:.3f} raw={s['raw']:.3f} lin={s['ZNE-lin']:.3f} quad={s['ZNE-quad']:.3f} exp0={s['ZNE-exp0']:.3f} exp3={s['ZNE-exp3']:.3f} | "
          f"quad boot/pred std {s['ZNE-quad_bootstd']:.4f}/{s['ZNE-quad_predstd']:.4f} | repstd quad {s['ZNE-quad_repstd']:.4f}")

# variance-amplification check: ratio of bootstrap std of ZNE to raw std, vs predicted sqrt(V_ZNE * K ...)
amp = []
for r in rows:
    for m in ["ZNE-lin", "ZNE-quad", "ZNE-exp0", "ZNE-exp3"]:
        amp.append(dict(method=m, emp=r[m + "_bootstd"] / r["raw_bootstd"], pred=r[m + "_predstd"] / r["raw_bootstd"]))
for m in ["ZNE-lin", "ZNE-quad", "ZNE-exp0", "ZNE-exp3"]:
    e = np.array([a["emp"] for a in amp if a["method"] == m]); p = np.array([a["pred"] for a in amp if a["method"] == m])
    print(m, "std amplification emp median %.2f pred median %.2f, ratio emp/pred median %.3f" % (np.median(e), np.median(p), np.median(e / p)))

# PEC table
pec_rows = []
for i, r in enumerate(pec["pec"]["results"]):
    raw = pec["raw"]["results"][i]
    key = (r["family"], r["n"], r["depth"]); E0 = ideal[key][0]
    g = r["metadata"]["resilience"]["layer_noise"]["noise_overhead"]
    pec_rows.append(dict(family=r["family"], n=r["n"], depth=r["depth"], E0=E0, raw_ev=raw["ev"], raw_std=raw["std"], raw_shots=raw["metadata"]["shots"],
                         pec_ev=r["ev"], pec_std=r["std"], pec_shots=r["metadata"]["shots"], gamma=g, gamma2=g * g,
                         pred_pec_std=float(np.sqrt(max(g * g - r["ev"] ** 2, 0) / r["metadata"]["shots"]))))
    print("PEC", key, "E0=%.3f raw=%.4f±%.4f pec=%.4f±%.4f gamma^2=%.2f pred std=%.4f" % (E0, raw["ev"], raw["std"], r["ev"], r["std"], g * g, pec_rows[-1]["pred_pec_std"]))
print("device seconds raw/pec:", pec["raw"]["metrics"]["usage"]["qpu_charge_time_seconds"], pec["pec"]["metrics"]["usage"]["qpu_charge_time_seconds"])

json.dump(dict(rows=rows, summary=summary, fit={str(k): v for k, v in fit.items()}, pec=pec_rows,
               device_seconds=dict(main_per_rep=60, raw_est=pec["raw"]["metrics"]["usage"]["qpu_charge_time_seconds"],
                                   pec_est=pec["pec"]["metrics"]["usage"]["qpu_charge_time_seconds"])),
          open(os.path.join(ROOT, "results", "hardware_summary.json"), "w"), indent=1, default=float)

# ---------------- Figure 7: hardware
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
ax = axes[0]
depths = [1, 2, 3, 4, 6, 8]
for s, mk in zip((1, 3, 5), ("o", "s", "^")):
    hw = [np.mean([r[f"E{s}"] for r in rows if r["family"] == "TFIM" and r["depth"] == d]) for d in depths]
    sd = [np.std([r[f"E{s}"] for r in rows if r["family"] == "TFIM" and r["depth"] == d], ddof=1) for d in depths]
    ax.errorbar(depths, hw, yerr=sd, fmt=mk, ms=4, capsize=2, label=rf"hardware $\lambda={s}$")
    ax.plot(depths, [fit[("TFIM", 6, d)]["pred"][s] for d in depths], ":", color=ax.lines[-1].get_color(), lw=1)
ax.plot(depths, [ideal[("TFIM", 6, d)][0] for d in depths], "k--", lw=1, label="ideal")
ax.set_xlabel("Trotter steps"); ax.set_ylabel(r"$\langle Z_3\rangle$"); ax.set_title("ibm_fez, 6 qubits", fontsize=9); ax.legend(frameon=False, fontsize=6.5)
ax = axes[1]
for m in ["raw", "ZNE-lin", "ZNE-quad", "ZNE-exp0", "ZNE-exp3"]:
    vals = [next(s for s in summary if s["family"] == "TFIM" and s["depth"] == d) for d in depths]
    ax.errorbar(np.array(depths) + {"raw": -0.2, "ZNE-lin": -0.1, "ZNE-quad": 0, "ZNE-exp0": 0.1, "ZNE-exp3": 0.2}[m],
                [v[m] - v["E0"] for v in vals], yerr=[v[m + "_bootstd"] for v in vals], fmt="o", ms=3, capsize=2, color=COL[m], label=m)
ax.axhline(0, color="k", lw=0.6); ax.set_xlabel("Trotter steps"); ax.set_ylabel("estimate $-$ ideal"); ax.set_title("mitigated estimates (3 repeats)", fontsize=9)
ax.legend(frameon=False, fontsize=6.5, ncol=2)
ax = axes[2]
for m in ["ZNE-lin", "ZNE-quad", "ZNE-exp0", "ZNE-exp3"]:
    e = [a["emp"] for a in amp if a["method"] == m]; p = [a["pred"] for a in amp if a["method"] == m]
    ax.plot(p, e, "o", ms=3, color=COL[m], label=m, alpha=0.8)
lim = [0.5, 8]; ax.plot(lim, lim, "k--", lw=0.8); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("predicted std amplification"); ax.set_ylabel("bootstrap std amplification"); ax.set_title("shot-noise amplification", fontsize=9)
ax.legend(frameon=False, fontsize=6.5)
fig.savefig(os.path.join(ROOT, "figures", "fig8_hardware.pdf")); plt.close(fig)
print("done")
