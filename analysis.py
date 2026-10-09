"""Analysis: bias/variance/MSE of all estimators, selection rule, MC validation, figures."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from estimators import *

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
COL = {"raw": "0.5", "ZNE-lin": "#1f77b4", "ZNE-quad": "#ff7f0e", "ZNE-exp0": "#2ca02c",
       "ZNE-exp3": "#9467bd", "PEC": "#d62728"}

rec = json.load(open("results/exact.json"))
os.makedirs("figures", exist_ok=True)
rng = np.random.default_rng(2026)
N_GRID = np.logspace(2, 8, 61)
summary = {}


def Evec(r, scales):
    return [r[f"E{s}"] for s in scales]


def zne_stats(r, method, N):
    scales, est = ZNE_METHODS[method]
    E = Evec(r, scales)
    val = est(E)
    bias = val - r["E0"]
    var, _ = delta_var(E, scales, est, N)
    return bias, var


def raw_stats(r, N):
    bias = r["E1"] - r["E0"]
    var = (1 - r["E1"] ** 2) / N
    return bias, var


def mse_all(r, N):
    out = {}
    b, v = raw_stats(r, N); out["raw"] = b * b + v
    for m in ZNE_METHODS:
        b, v = zne_stats(r, m, N); out[m] = b * b + v
    mse, b, v = pec_mse(r["gamma"], r["E0"], r["E0"], N); out["PEC"] = mse
    return out


def kappa(r):
    """Decay rate per unit scale from the pilot data (l = 1, 3)."""
    return -np.log(abs(r["E3"] / r["E1"])) / 2


def n_star(r, method):
    """Shot budget above which PEC beats the ZNE method (analytic)."""
    scales, est = ZNE_METHODS[method]
    E = Evec(r, scales)
    bias = est(E) - r["E0"]
    var1, _ = delta_var(E, scales, est, 1.0)  # variance * N
    num = r["gamma"] ** 2 - r["E0"] ** 2 - var1
    if bias == 0:
        return np.inf if num > 0 else 0.0
    if num <= 0:
        return 0.0
    return num / bias ** 2


def get(fam, **kw):
    out = [r for r in rec if r["family"] == fam and all(abs(r[k] - v) < 1e-12 for k, v in kw.items())]
    return out


# ------------------------------------------------------------------ Fig 1: E(lambda)
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
for ax, (fam, kw, title) in zip(axes, [("TFIM", dict(depth=5), "TFIM, 5 Trotter steps"),
                                        ("HEA", dict(depth=2), "HEA, 2 layers"),
                                        ("GHZ", dict(n=6), "GHZ, 6 qubits")]):
    for p2, mk in zip([0.005, 0.01, 0.02], ["o", "s", "^"]):
        r = get(fam, p2=p2, **kw)[0]
        ls = np.array([0, 1, 3, 5])
        ax.plot(ls, [r["E0"], r["E1"], r["E3"], r["E5"]], marker=mk, ms=4, lw=1,
                label=f"$p_2={p2}$")
        # quadratic extrapolation line
        E = Evec(r, SCALES3)
        c = np.polyfit(SCALES3, E, 2)
        xx = np.linspace(0, 5, 50)
        ax.plot(xx, np.polyval(c, xx), ":", lw=0.8, color=ax.lines[-1].get_color())
    ax.axhline(r["E0"], color="k", lw=0.6, ls="--")
    ax.set_xlabel(r"noise scale $\lambda$"); ax.set_title(title, fontsize=9)
    ax.set_xticks([0, 1, 3, 5])
axes[0].set_ylabel(r"$\langle O\rangle_\lambda$"); axes[0].legend(frameon=False)
fig.savefig("figures/fig1_scaling.pdf"); plt.close(fig)

# ------------------------------------------------------------------ Fig 2: MSE vs N (+MC)
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
mc_check = {}
for ax, (fam, kw, title) in zip(axes, [("TFIM", dict(depth=5, p2=0.01), "TFIM d=5, $p_2$=0.01"),
                                        ("TFIM", dict(depth=10, p2=0.01), "TFIM d=10, $p_2$=0.01"),
                                        ("GHZ", dict(n=6, p2=0.01), "GHZ n=6, $p_2$=0.01")]):
    r = get(fam, **kw)[0]
    curves = {m: [] for m in ["raw", "PEC"] + list(ZNE_METHODS)}
    for N in N_GRID:
        for m, v in mse_all(r, N).items():
            curves[m].append(v)
    for m, v in curves.items():
        ax.loglog(N_GRID, v, color=COL[m], lw=1.2, label=m)
    # Monte Carlo validation points
    for N in [1e3, 1e4, 1e5, 1e6]:
        for m in ["ZNE-lin", "ZNE-quad", "ZNE-exp0", "ZNE-exp3"]:
            scales, est = ZNE_METHODS[m]
            s = shot_mc(Evec(r, scales), scales, est, N, 4000, rng)
            mse = np.mean((s - r["E0"]) ** 2)
            ax.plot(N, mse, "o", ms=3, mfc="none", color=COL[m])
            mc_check[f"{fam}-{kw}-{m}-{int(N)}"] = (mse, float(np.interp(N, N_GRID, curves[m])))
        # PEC MC: estimator = gamma * s where s in {+-1}, P(s=+1) = (1 + E0/gamma)/2
        pplus = (1 + r["E0"] / r["gamma"]) / 2
        ks = rng.binomial(int(N), pplus, size=4000)
        est = r["gamma"] * (2 * ks / int(N) - 1)
        ax.plot(N, np.mean((est - r["E0"]) ** 2), "o", ms=3, mfc="none", color=COL["PEC"])
    ax.set_title(title, fontsize=9); ax.set_xlabel("total shots $N$")
    ax.set_ylim(1e-9, 10)
    ax.text(0.03, 0.04, rf"$\gamma^2={r['gamma']**2:.3g}$", transform=ax.transAxes, fontsize=7.5)
axes[0].set_ylabel("MSE"); axes[0].legend(frameon=False, ncol=2, loc="upper right")
fig.savefig("figures/fig2_mse_vs_N.pdf"); plt.close(fig)
summary["mc_check"] = mc_check

# ------------------------------------------------------------------ Fig 3: phase diagram (TFIM)
depths = sorted(set(r["depth"] for r in rec if r["family"] == "TFIM"))
methods = ["raw", "ZNE-lin", "ZNE-quad", "ZNE-exp0", "ZNE-exp3", "PEC"]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
for ax, p2 in zip(axes, [0.005, 0.01, 0.02]):
    best = np.zeros((len(N_GRID), len(depths)), int)
    nstar_q = []
    for j, d in enumerate(depths):
        r = get("TFIM", depth=d, p2=p2)[0]
        for i, N in enumerate(N_GRID):
            m = mse_all(r, N)
            best[i, j] = methods.index(min(m, key=m.get))
        nstar_q.append(n_star(r, "ZNE-quad"))
    cmap = matplotlib.colors.ListedColormap([COL[m] for m in methods])
    ax.pcolormesh(np.array(depths), N_GRID, best, cmap=cmap, vmin=-0.5, vmax=len(methods) - 0.5,
                  shading="nearest")
    ax.plot(depths, nstar_q, "k-", lw=1.2, label=r"$N^*$ (PEC vs ZNE-quad)")
    ax.set_yscale("log"); ax.set_xlabel("Trotter steps"); ax.set_title(f"$p_2={p2}$", fontsize=9)
    ax.set_ylim(N_GRID[0], N_GRID[-1])
axes[0].set_ylabel("total shots $N$")
handles = [plt.Rectangle((0, 0), 1, 1, color=COL[m]) for m in methods]
axes[2].legend(handles + [plt.Line2D([], [], color="k")], methods + [r"$N^*$"], frameon=True,
               fontsize=6.5, loc="upper left", bbox_to_anchor=(1.02, 1.0))
fig.savefig("figures/fig3_phase.pdf"); plt.close(fig)

# ------------------------------------------------------------------ Fig 4: gamma^2 vs ZNE variance factor; N*
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
ax = axes[0]
mk = {"TFIM": "o", "HEA": "s", "GHZ": "^"}
for fam in ["TFIM", "HEA", "GHZ"]:
    rs = [r for r in rec if r["family"] == fam]
    eps = np.array([r["eps"] for r in rs])
    ax.semilogy(eps, [r["gamma"] ** 2 for r in rs], mk[fam], ms=3.5, mfc="none", color=COL["PEC"], label=rf"$\gamma^2$, {fam}")
ee = np.linspace(0, 2.1, 50)
ax.semilogy(ee, np.exp(4 * ee), "k--", lw=0.8, label=r"$e^{4\epsilon}$")
VL = 2 * np.sum(C_LIN ** 2); VQ = 3 * np.sum(C_QUAD ** 2)
ax.axhline(VL, color=COL["ZNE-lin"], lw=1); ax.axhline(VQ, color=COL["ZNE-quad"], lw=1)
ax.text(0.02, VL * 1.15, r"$V_{\rm lin}=5$", color=COL["ZNE-lin"], fontsize=7)
ax.text(0.02, VQ * 1.15, r"$V_{\rm quad}=15.7$", color=COL["ZNE-quad"], fontsize=7)
ax.axvline(np.log(VL) / 4, color=COL["ZNE-lin"], lw=0.6, ls=":"); ax.axvline(np.log(VQ) / 4, color=COL["ZNE-quad"], lw=0.6, ls=":")
ax.set_xlabel(r"expected error count $\epsilon = n_1 p_1 + n_2 p_2$"); ax.set_ylabel("shot-variance factor")
ax.set_ylim(0.8, 1e4); ax.set_xlim(0, 2.1); ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(0.25, 1.0))
ax.set_title("PEC overhead vs ZNE variance amplification", fontsize=9)

ax = axes[1]
pred_err = []
for fam in ["TFIM", "HEA", "GHZ"]:
    rs = [r for r in rec if r["family"] == fam]
    for m, col in [("ZNE-lin", COL["ZNE-lin"]), ("ZNE-quad", COL["ZNE-quad"])]:
        eps = np.array([r["eps"] for r in rs]); ns = np.array([n_star(r, m) for r in rs])
        sel = ns > 0
        ax.semilogy(eps[sel], ns[sel], mk[fam], ms=3.5, mfc="none", color=col,
                    label=f"{m}, {fam}" if fam == "TFIM" else None)
    for r in rs:
        nn = n_star(r, "ZNE-quad")
        if nn > 0 and abs(r["E0"]) >= 0.2:
            k = kappa(r); b_pred = 2.5 * k ** 3 * abs(r["E0"])
            E = Evec(r, SCALES3); VZ = 3 * np.sum(C_QUAD ** 2 * (1 - np.array(E) ** 2))
            n_pred = (r["gamma"] ** 2 - r["E0"] ** 2 - VZ) / b_pred ** 2
            if n_pred > 0:
                pred_err.append(np.log10(n_pred / nn))
                ax.plot([r["eps"]], [n_pred], "x", ms=3, color=COL["ZNE-quad"])
ax.set_xlabel(r"$\epsilon$"); ax.set_ylabel(r"$N^*$ (PEC preferred above)")
ax.set_title(r"Crossover budget where $N^*>0$ (x: analytic)", fontsize=9)
ax.legend(frameon=False, fontsize=6.5); ax.set_xlim(0, 2.1)
summary["nstar_pred_log10_err_median"] = float(np.median(pred_err)) if pred_err else None
summary["nstar_pred_log10_err_iqr"] = [float(np.percentile(pred_err, 25)), float(np.percentile(pred_err, 75))] if pred_err else None
summary["nstar_pred_n"] = len(pred_err)
fig.savefig("figures/fig4_nstar.pdf"); plt.close(fig)

# ------------------------------------------------------------------ Fig 5: miscalibration
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
deltas = sorted(float(d) for d in rec[0]["miscal"].keys())
for ax, N in zip(axes, [1e4, 1e6]):
    for fam, kw, lab in [("TFIM", dict(depth=5, p2=0.01), "TFIM d=5"),
                         ("TFIM", dict(depth=10, p2=0.01), "TFIM d=10"),
                         ("GHZ", dict(n=6, p2=0.01), "GHZ n=6")]:
        r = get(fam, **kw)[0]
        mses = []
        for d in deltas:
            mm = r["miscal"][str(d)]
            mses.append(pec_mse(mm["gamma_hat"], mm["E_target"], r["E0"], N)[0])
        l, = ax.semilogy(deltas, mses, "o-", ms=3, lw=1, label=f"PEC, {lab}")
        bq = zne_stats(r, "ZNE-quad", N); be = zne_stats(r, "ZNE-exp3", N)
        best_z = min(bq[0] ** 2 + bq[1], be[0] ** 2 + be[1])
        ax.axhline(best_z, color=l.get_color(), ls="--", lw=0.8)
    ax.set_xlabel(r"relative calibration error $\delta$"); ax.set_title(f"$N=10^{int(np.log10(N))}$", fontsize=9)
axes[0].set_ylabel("MSE"); axes[0].legend(frameon=False, fontsize=6.5)
axes[1].text(0.02, 0.95, "dashed: best ZNE (quad/exp3)", transform=axes[1].transAxes, fontsize=7, va="top")
fig.savefig("figures/fig5_miscal.pdf"); plt.close(fig)

# ------------------------------------------------------------------ tables / numbers for text
tab = []
for fam, kw in [("TFIM", dict(depth=5, p2=0.01)), ("TFIM", dict(depth=10, p2=0.01)),
                ("HEA", dict(depth=4, p2=0.01)), ("GHZ", dict(n=6, p2=0.01)), ("GHZ", dict(n=8, p2=0.01))]:
    r = get(fam, **kw)[0]
    row = dict(family=fam, **kw, n1=r["n1"], n2=r["n2"], eps=r["eps"], gamma2=r["gamma"] ** 2,
               E0=r["E0"], E1=r["E1"], kappa=kappa(r))
    for m in ZNE_METHODS:
        b, v1 = zne_stats(r, m, 1.0)
        row[f"{m}_bias"] = b; row[f"{m}_varN"] = v1
        row[f"{m}_Nstar"] = n_star(r, m)
    for N in [1e3, 1e4, 1e5, 1e6]:
        m = mse_all(r, N); row[f"best@{int(N)}"] = min(m, key=m.get)
    tab.append(row)
summary["table"] = tab

# delta at which PEC MSE equals best ZNE MSE (TFIM d=5, N=1e6)
def pec_mis_mse(r, d, N):
    mm = r["miscal"][str(d)]
    return pec_mse(mm["gamma_hat"], mm["E_target"], r["E0"], N)[0]

crit = {}
for fam, kw in [("TFIM", dict(depth=5, p2=0.01)), ("TFIM", dict(depth=10, p2=0.01)), ("GHZ", dict(n=6, p2=0.01))]:
    r = get(fam, **kw)[0]
    for N in [1e4, 1e6]:
        bq = zne_stats(r, "ZNE-quad", N); be = zne_stats(r, "ZNE-exp3", N)
        best_z = min(bq[0] ** 2 + bq[1], be[0] ** 2 + be[1])
        ok = [d for d in deltas if pec_mis_mse(r, d, N) < best_z]
        crit[f"{fam}-{kw}-N{int(N)}"] = dict(deltas_where_pec_wins=ok, best_zne_mse=best_z,
                                             pec_exact_mse=pec_mse(r["gamma"], r["E0"], r["E0"], N)[0])
summary["miscal"] = crit

# fraction of configurations where PEC is best at each N
frac = {}
for N in [1e3, 1e4, 1e5, 1e6, 1e7]:
    wins = {}
    for r in rec:
        m = mse_all(r, N); b = min(m, key=m.get); wins[b] = wins.get(b, 0) + 1
    frac[int(N)] = wins
summary["best_counts"] = frac

# how often ZNE-exp0 beats ZNE-quad (bias) and raw improvement
summary["n_records"] = len(rec)
json.dump(summary, open("results/summary.json", "w"), indent=1, default=float)
print(json.dumps(summary["table"], indent=1, default=float)[:6000])
print(json.dumps(summary["best_counts"], indent=1))
print("miscal:", json.dumps(summary["miscal"], indent=1, default=float))
print("nstar pred err median log10:", summary["nstar_pred_log10_err_median"], summary["nstar_pred_log10_err_iqr"])

# ================================================================== extra: rule regret & delta*(N)
from qsim import DensityMatrixSim, lam_from_p, gamma_from_lam
from circuits import *

def practical_rule(r, N):
    """Decision using only pilot data (E1,E3,E5 noisy-exact here) and gamma."""
    scales, est = ZNE_METHODS["ZNE-quad"]
    E = Evec(r, scales)
    VZ = 3 * np.sum(C_QUAD ** 2 * (1 - np.array(E) ** 2))
    g2 = r["gamma"] ** 2
    if g2 <= VZ:
        return "PEC"
    E0h = zne_exp_zero(E[:2])
    if not np.isfinite(E0h) or abs(E0h) < 1e-3:
        return "PEC" if N > g2 else "ZNE-quad"
    k = kappa(r)
    b = 2.5 * k ** 3 * abs(E0h)
    nstar = (g2 - E0h ** 2 - VZ) / b ** 2
    return "PEC" if N > nstar else "ZNE-quad"

reg_all = []
for r in rec:
    for N in np.logspace(2, 7, 26):
        m = mse_all(r, N)
        choice = practical_rule(r, N)
        oracle2 = min(m["PEC"], m["ZNE-quad"])
        reg_all.append((m[choice] / oracle2, m[choice] / min(m.values()), r["family"], N))
reg = np.array([x[0] for x in reg_all]); reg_full = np.array([x[1] for x in reg_all])
summary["rule_regret_vs_pec_quad"] = dict(median=float(np.median(reg)), p90=float(np.percentile(reg, 90)),
                                          max=float(reg.max()), frac_gt_2=float(np.mean(reg > 2)))
summary["rule_regret_vs_all_methods"] = dict(median=float(np.median(reg_full)), p90=float(np.percentile(reg_full, 90)),
                                             max=float(reg_full.max()), frac_gt_2=float(np.mean(reg_full > 2)))
# always-PEC baseline regret
reg_pec = np.array([mse_all(r, N)["PEC"] / min(mse_all(r, N).values()) for r in rec for N in np.logspace(2, 7, 26)])
summary["always_pec_regret"] = dict(median=float(np.median(reg_pec)), p90=float(np.percentile(reg_pec, 90)),
                                    max=float(reg_pec.max()), frac_gt_2=float(np.mean(reg_pec > 2)))
reg_q = np.array([mse_all(r, N)["ZNE-quad"] / min(mse_all(r, N).values()) for r in rec for N in np.logspace(2, 7, 26)])
summary["always_znequad_regret"] = dict(median=float(np.median(reg_q)), p90=float(np.percentile(reg_q, 90)),
                                        max=float(reg_q.max()), frac_gt_2=float(np.mean(reg_q > 2)))

# bias constant check restricted to |E0| >= 0.2
ratios = []
for r in rec:
    if abs(r["E0"]) >= 0.2 and r["family"] != "GHZ":
        k = kappa(r); b = zne_stats(r, "ZNE-quad", 1)[0]
        ratios.append(b / (k ** 3 * r["E0"]))
summary["bias_const_ratio"] = dict(median=float(np.median(ratios)), p10=float(np.percentile(ratios, 10)),
                                   p90=float(np.percentile(ratios, 90)), n=len(ratios))

# epsilon thresholds where gamma^2 = variance factor
summary["eps_threshold_lin"] = float(np.log(2 * np.sum(C_LIN ** 2)) / 4)
summary["eps_threshold_quad"] = float(np.log(3 * np.sum(C_QUAD ** 2)) / 4)
summary["varfac_lin"] = float(2 * np.sum(C_LIN ** 2)); summary["varfac_quad"] = float(3 * np.sum(C_QUAD ** 2))

# delta*(N): critical relative calibration error for PEC to still beat best ZNE
def delta_star_curve(fam, kw, Ns):
    r = get(fam, **kw)[0]
    n = r["n"]
    circ, O = {"TFIM": lambda: (tfim_trotter(n, r["depth"]), tfim_observable(n)[0]),
               "GHZ": lambda: (ghz(n), ghz_observable(n)[0]),
               "HEA": lambda: (hea(n, r["depth"]), hea_observable(n)[0])}[fam]()
    dm = DensityMatrixSim(n)
    p1, p2 = r["p1"], r["p2"]
    l1, l2 = lam_from_p(p1, 1), lam_from_p(p2, 2)
    dgrid = np.linspace(0.0, 0.5, 201)[1:]
    Et, gh = [], []
    for d in dgrid:
        l1h, l2h = lam_from_p(p1 * (1 + d), 1), lam_from_p(p2 * (1 + d), 2)
        Et.append(dm.expect(dm.run(circ, l1 / l1h, l2 / l2h), O))
        gh.append(gamma_from_lam(l1h, 1) ** r["n1"] * gamma_from_lam(l2h, 2) ** r["n2"])
    Et, gh = np.array(Et), np.array(gh)
    out = []
    for N in Ns:
        bq = zne_stats(r, "ZNE-quad", N); be = zne_stats(r, "ZNE-exp3", N); b0 = zne_stats(r, "ZNE-exp0", N)
        best_z = min(bq[0] ** 2 + bq[1], be[0] ** 2 + be[1], b0[0] ** 2 + b0[1])
        mses = (Et - r["E0"]) ** 2 + (gh ** 2 - Et ** 2) / N
        ok = dgrid[mses < best_z]
        out.append(ok.max() if len(ok) else 0.0)
    return np.array(out), r

Ns = np.logspace(2, 8, 25)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
ax = axes[0]
for fam, kw, lab in [("TFIM", dict(depth=5, p2=0.01), "TFIM d=5"), ("TFIM", dict(depth=10, p2=0.01), "TFIM d=10"),
                     ("GHZ", dict(n=6, p2=0.01), "GHZ n=6"), ("HEA", dict(depth=4, p2=0.01), "HEA L=4")]:
    r = get(fam, **kw)[0]
    mses = []
    for d in deltas:
        mm = r["miscal"][str(d)]
        mses.append(pec_mse(mm["gamma_hat"], mm["E_target"], r["E0"], 1e5)[0])
    l, = ax.semilogy(deltas, mses, "o-", ms=3, lw=1, label=f"PEC, {lab}")
    bq = zne_stats(r, "ZNE-quad", 1e5); be = zne_stats(r, "ZNE-exp3", 1e5); b0 = zne_stats(r, "ZNE-exp0", 1e5)
    ax.axhline(min(bq[0] ** 2 + bq[1], be[0] ** 2 + be[1], b0[0] ** 2 + b0[1]), color=l.get_color(), ls="--", lw=0.8)
ax.set_xlabel(r"relative calibration error $\delta$"); ax.set_ylabel("MSE"); ax.set_title("$N=10^5$; dashed: best ZNE", fontsize=9)
ax.legend(frameon=False, fontsize=6.5)
ax = axes[1]
dstar_store = {}
for fam, kw, lab in [("TFIM", dict(depth=5, p2=0.01), "TFIM d=5"), ("TFIM", dict(depth=10, p2=0.01), "TFIM d=10"),
                     ("GHZ", dict(n=6, p2=0.01), "GHZ n=6"), ("HEA", dict(depth=4, p2=0.01), "HEA L=4")]:
    ds, r = delta_star_curve(fam, kw, Ns)
    ax.loglog(Ns, np.maximum(ds, 1e-3), "-", lw=1.2, label=lab)
    dstar_store[lab] = dict(N=list(Ns), delta_star=list(ds))
# scaling guide: delta* ~ N^{-1/2}
ax.loglog(Ns, 0.5 * (Ns / 1e3) ** -0.5, "k:", lw=0.8, label=r"$\propto N^{-1/2}$")
ax.set_xlabel("total shots $N$"); ax.set_ylabel(r"tolerable $|\delta|$ for PEC to win")
ax.set_ylim(1e-3, 0.6); ax.legend(frameon=False, fontsize=6.5)
fig.savefig("figures/fig5_miscal.pdf"); plt.close(fig)
summary["delta_star"] = dstar_store

json.dump(summary, open("results/summary.json", "w"), indent=1, default=float)
for k in ["rule_regret_vs_pec_quad", "rule_regret_vs_all_methods", "always_pec_regret", "always_znequad_regret",
          "bias_const_ratio", "eps_threshold_lin", "eps_threshold_quad", "varfac_lin", "varfac_quad"]:
    print(k, summary[k])
for k, v in dstar_store.items():
    print(k, [f"{n:.0e}:{d:.3f}" for n, d in zip(v["N"][::4], v["delta_star"][::4])])
