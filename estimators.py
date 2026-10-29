"""ZNE extrapolators and PEC cost model on top of exact noisy expectation values."""
import numpy as np

SCALES3 = np.array([1, 3, 5])
SCALES2 = np.array([1, 3])


def richardson_coeffs(scales):
    """Coefficients c_l such that sum c_l f(l) is the polynomial extrapolation to 0."""
    s = np.asarray(scales, float)
    c = np.ones_like(s)
    for i in range(len(s)):
        for j in range(len(s)):
            if i != j:
                c[i] *= s[j] / (s[j] - s[i])
    return c


C_LIN = richardson_coeffs(SCALES2)   # [1.5, -0.5]
C_QUAD = richardson_coeffs(SCALES3)  # [15/8, -5/4, 3/8]


def zne_linear(E):   # E = [E1, E3]
    return C_LIN @ np.asarray(E)


def zne_quadratic(E):  # E = [E1, E3, E5]
    return C_QUAD @ np.asarray(E)


def zne_exp_zero(E):
    """Exponential fit with known asymptote 0 through (1,E1),(3,E3):
    E(l) = B r^l  ->  E(0) = E1 * (E1/E3)^{1/2}.  Requires E1/E3 > 0."""
    E1, E3 = float(E[0]), float(E[1])
    if E3 == 0:
        return np.nan
    ratio = E1 / E3
    with np.errstate(invalid="ignore", divide="ignore"):
        out = E1 * np.sqrt(np.abs(ratio)) * np.sign(ratio)
    return out


def zne_exp_three(E):
    """Three-point geometric fit E(l) = A + B r^l through l=1,3,5 (equally spaced).
    A = (E1 E5 - E3^2) / (E1 + E5 - 2 E3); E(0) = A + (E1-A) (E1-A)/(E3-A) ... with
    step 2 in l, r^2 = (E3-A)/(E1-A), so E(0) = A + (E1-A) * r^{-1}."""
    E1, E3, E5 = float(E[0]), float(E[1]), float(E[2])
    den = E1 + E5 - 2 * E3
    if den == 0:
        return np.nan
    A = (E1 * E5 - E3 ** 2) / den
    if E1 == A or E3 == A:
        return np.nan
    r2 = (E3 - A) / (E1 - A)
    if r2 <= 0:
        return np.nan  # non-monotone data: geometric model not applicable
    return A + (E1 - A) / np.sqrt(r2)


ZNE_METHODS = {
    "ZNE-lin": (SCALES2, zne_linear),
    "ZNE-quad": (SCALES3, zne_quadratic),
    "ZNE-exp0": (SCALES2, zne_exp_zero),
    "ZNE-exp3": (SCALES3, zne_exp_three),
}


def shot_mc(E_exact, scales, estimator, N_total, reps, rng):
    """Monte Carlo of the finite-shot ZNE estimator.  Each scale gets
    N_total/len(scales) shots of a +-1 observable with mean E_exact[l]."""
    K = len(scales)
    N = max(int(N_total // K), 1)
    out = np.empty(reps)
    for r in range(reps):
        Eh = []
        for e in E_exact:
            pplus = (1 + e) / 2
            k = rng.binomial(N, pplus)
            Eh.append(2 * k / N - 1)
        out[r] = estimator(Eh)
    out = out[np.isfinite(out)]
    return out


def delta_var(E_exact, scales, estimator, N_total, h=1e-5):
    """Delta-method variance: grad^T Sigma grad with Sigma = diag((1-E^2)/(N/K))."""
    K = len(scales)
    N = N_total / K
    E = np.asarray(E_exact, float)
    g = np.zeros_like(E)
    for i in range(len(E)):
        Ep = E.copy(); Ep[i] += h
        Em = E.copy(); Em[i] -= h
        g[i] = (estimator(Ep) - estimator(Em)) / (2 * h)
    var = np.sum(g ** 2 * (1 - E ** 2) / N)
    return var, g


def pec_mse(gamma, E_target, E_true, N):
    """PEC mean squared error with total overhead gamma.  E_target is the value the
    (possibly miscalibrated) PEC estimator converges to; E_true the ideal value.
    Single-shot estimator = gamma * sign * (+-1) so second moment = gamma^2."""
    bias = E_target - E_true
    var = (gamma ** 2 - E_target ** 2) / N
    return bias ** 2 + var, bias, var
