import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from estimators import *

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5, "figure.dpi": 150, "savefig.bbox": "tight"})
COL = {"raw": "0.5", "ZNE-lin": "#1f77b4", "ZNE-quad": "#ff7f0e", "ZNE-exp0": "#2ca02c", "ZNE-exp3": "#9467bd", "PEC": "#d62728"}
D = json.load(open("results/ampdamp.json")); recs = D["records"]
N_GRID = np.logspace(2, 8, 61)
out = {}

def stats(r, pref, N):
    """MSE dict for scenario prefix 'tw' or 'ad'."""
    E = {s: r[f"{pref}_E{s}"] for s in (1, 3, 5)}
    m = {}
    m["raw"] = (E[1] - r["E0"]) ** 2 + (1 - E[1] ** 2) / N
    for name, (sc, est) in ZNE_METHODS.items():
        Ev = [E[s] for s in sc]; b = est(Ev) - r["E0"]
        v, _ = delta_var(Ev, sc, est, N)
        m[name] = (b * b + v) if np.isfinite(b) else np.inf
    if pref == "tw":
        m["PEC"] = (r["gamma"] ** 2 - r["E0"] ** 2) / N
    else:
        t = r["ad_pec_target"]; m["PEC"] = (t - r["E0"]) ** 2 + (r["gamma"] ** 2 - t ** 2) / N
    return m

def biases(r, pref):
    E = {s: r[f"{pref}_E{s}"] for s in (1, 3, 5)}
    b = {name: est([E[s] for s in sc]) - r["E0"] for name, (sc, est) in ZNE_METHODS.items()}
    b["PEC"] = 0.0 if pref == "tw" else r["ad_pec_target"] - r["E0"]
    b["raw"] = E[1] - r["E0"]
    return b

methods = ["raw", "ZNE-lin", "ZNE-quad", "ZNE-exp0", "ZNE-exp3", "PEC"]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
# (a) scaling curves d=5
r = recs[4]; ax = axes[0]
ls = [0, 1, 3, 5]
ax.plot(ls, [r["E0"]] + [r[f"tw_E{s}"] for s in (1, 3, 5)], "o-", ms=4, lw=1, label="twirled (Pauli)")
ax.plot(ls, [r["E0"]] + [r[f"ad_E{s}"] for s in (1, 3, 5)], "s-", ms=4, lw=1, label="amplitude damping")
ax.axhline(r["E0"], color="k", ls="--", lw=0.6); ax.set_xticks(ls)
ax.set_xlabel(r"noise scale $\lambda$"); ax.set_ylabel(r"$\langle Z_3\rangle_\lambda$"); ax.set_title("TFIM d=5", fontsize=9)
ax.legend(frameon=False)
# (b) MSE vs N, untwirled, d=5
ax = axes[1]
for m in methods:
    ax.loglog(N_GRID, [stats(r, "ad", N)[m] for N in N_GRID], color=COL[m], lw=1.2, label=m)
ax.loglog(N_GRID, [stats(r, "tw", N)["PEC"] for N in N_GRID], color=COL["PEC"], lw=1, ls=":", label="PEC (twirled)")
ax.set_xlabel("total shots $N$"); ax.set_ylabel("MSE"); ax.set_title("untwirled, d=5", fontsize=9)
ax.set_ylim(1e-9, 1e3); ax.legend(frameon=False, ncol=2, fontsize=5.5, loc="upper right")
# (c) best method, untwirled, depth x N
ax = axes[2]
depths = [r["depth"] for r in recs]
best = np.zeros((len(N_GRID), len(depths)), int)
for j, r in enumerate(recs):
    for i, N in enumerate(N_GRID):
        m = stats(r, "ad", N); best[i, j] = methods.index(min(m, key=m.get))
cmap = matplotlib.colors.ListedColormap([COL[m] for m in methods])
ax.pcolormesh(np.array(depths), N_GRID, best, cmap=cmap, vmin=-0.5, vmax=len(methods) - 0.5, shading="nearest")
ax.set_yscale("log"); ax.set_xlabel("Trotter steps"); ax.set_title("optimal method, untwirled", fontsize=9)
ax.set_ylim(N_GRID[0], N_GRID[-1])
fig.savefig("figures/fig6_ampdamp.pdf"); plt.close(fig)

# numbers
for pref in ("tw", "ad"):
    cnt = {}
    for N in [1e3, 1e4, 1e5, 1e6, 1e7]:
        c = {}
        for r in recs:
            m = stats(r, pref, N); b = min(m, key=m.get); c[b] = c.get(b, 0) + 1
        cnt[int(N)] = c
    out[f"best_counts_{pref}"] = cnt
out["d5_biases_ad"] = biases(recs[4], "ad"); out["d5_biases_tw"] = biases(recs[4], "tw")
out["d10_biases_ad"] = biases(recs[9], "ad")
out["pec_bias_ad_all"] = [(r["depth"], r["ad_pec_target"] - r["E0"], r["gamma"] ** 2, r["eps"]) for r in recs]
# thresholds: gamma^2 vs 15.66 -> depth at which PEC no longer dominates quad
out["gamma2_by_depth"] = [(r["depth"], r["gamma"] ** 2) for r in recs]
# N at which ZNE-exp3 overtakes PEC(ad) for d=5
for d_idx in (4, 9):
    r = recs[d_idx]
    cross = [N for N in N_GRID if stats(r, "ad", N)["ZNE-exp3"] < stats(r, "ad", N)["PEC"]]
    out[f"exp3_beats_pec_ad_from_N_d{r['depth']}"] = (min(cross) if cross else None)
    cross = [N for N in N_GRID if stats(r, "ad", N)["ZNE-quad"] < stats(r, "ad", N)["PEC"]]
    out[f"quad_beats_pec_ad_from_N_d{r['depth']}"] = (min(cross) if cross else None)
json.dump(out, open("results/ampdamp_summary.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
