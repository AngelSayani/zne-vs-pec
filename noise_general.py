"""General Pauli-channel tools and Kraus-sum density-matrix evolution.

Used for the amplitude-damping robustness study:
  * physical noise after a k-qubit gate = k-qubit depolarizing (p_k) followed by
    single-qubit amplitude damping (rate g_k) on each of the gate's qubits;
  * 'twirled' variant: amplitude damping replaced by its Pauli twirl (what
    randomized compiling produces), so the total noise is a Pauli channel;
  * PEC decomposition always built from the Pauli (twirled) model.
"""
import itertools
import numpy as np
from qsim import I2, X, Y, Z, PAULIS1

# ---------------------------------------------------------------- Pauli algebra on k qubits
def pauli_ops(k):
    ops = []
    for idx in itertools.product(range(4), repeat=k):
        m = np.array([[1.0 + 0j]])
        for i in idx:
            m = np.kron(m, PAULIS1[i])
        ops.append(m)
    return ops  # identity first (idx all zeros)


def commutation_sign_matrix(k):
    """chi[P,Q] = +1 if P,Q commute, -1 if anticommute (single-qubit factors multiply)."""
    d = 4 ** k
    # single-qubit: I commutes with all; X,Y,Z anticommute pairwise (distinct non-identity)
    chi1 = np.ones((4, 4))
    for a in range(1, 4):
        for b in range(1, 4):
            if a != b:
                chi1[a, b] = -1
    chi = np.array([[1.0]])
    for _ in range(k):
        chi = np.kron(chi, chi1)
    return chi


def pauli_probs_to_eigs(probs, k):
    """Pauli channel with probabilities probs[P] has eigenvalue lam[Q] = sum_P probs[P] chi[P,Q]
    on Pauli Q (Q -> lam[Q] Q)."""
    chi = commutation_sign_matrix(k)
    return probs @ chi


def eigs_to_pauli_coeffs(lams, k):
    """Inverse transform: coefficients a[P] with sum_P a[P] P rho P having eigenvalues lams."""
    chi = commutation_sign_matrix(k)
    return (chi @ lams) / (4 ** k)


def inverse_pauli_channel(probs, k):
    """Quasi-probability of the inverse of a Pauli channel and its 1-norm gamma."""
    lams = pauli_probs_to_eigs(probs, k)
    a = eigs_to_pauli_coeffs(1.0 / lams, k)
    return a, float(np.abs(a).sum())


def compose_pauli(probs_a, probs_b, k):
    """Composition of two Pauli channels (probability vectors) via eigenvalue product."""
    la = pauli_probs_to_eigs(probs_a, k)
    lb = pauli_probs_to_eigs(probs_b, k)
    return eigs_to_pauli_coeffs(la * lb, k)


def depolarizing_probs(p, k):
    d = 4 ** k
    v = np.full(d, p / (d - 1)); v[0] = 1 - p
    return v


def amp_damp_kraus(g):
    K0 = np.array([[1, 0], [0, np.sqrt(1 - g)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(g)], [0, 0]], dtype=complex)
    return [K0, K1]


def amp_damp_twirl_probs(g):
    """Pauli twirl of single-qubit amplitude damping with damping probability g."""
    s = np.sqrt(1 - g)
    pX = pY = g / 4
    pZ = (1 - g / 2 - s) / 2
    pI = 1 - pX - pY - pZ
    return np.array([pI, pX, pY, pZ])


def tensor_probs(p_list):
    out = np.array([1.0])
    for p in p_list:
        out = np.kron(out, p)
    return out


# ---------------------------------------------------------------- Kraus-sum evolution
class KrausDensityMatrixSim:
    """Density-matrix simulation where each noise element is a list of (coef, K) terms:
    rho -> sum_i coef_i K_i rho K_i^dagger  (coef may be negative: quasi-channels)."""

    def __init__(self, n):
        self.n = n; self.dim = 2 ** n

    def zero_state(self):
        rho = np.zeros((self.dim, self.dim), dtype=complex); rho[0, 0] = 1; return rho

    def _apply_op(self, rho, K, qubits):
        n, k = self.n, len(qubits)
        t = rho.reshape([2] * (2 * n)); Kt = K.reshape([2] * (2 * k))
        t = np.tensordot(Kt, t, axes=(list(range(k, 2 * k)), list(qubits)))
        t = np.moveaxis(t, list(range(k)), list(qubits))
        col = [n + q for q in qubits]
        t = np.tensordot(t, Kt.conj(), axes=(col, list(range(k, 2 * k))))
        t = np.moveaxis(t, list(range(2 * n - k, 2 * n)), col)
        return t.reshape(self.dim, self.dim)

    def apply_terms(self, rho, terms, qubits):
        out = np.zeros_like(rho)
        for c, K in terms:
            if c != 0:
                out += c * self._apply_op(rho, K, qubits)
        return out

    def run(self, circuit, noise_fn, rho=None):
        """noise_fn(gate) -> list of (terms, qubits) to apply in order after the gate."""
        if rho is None:
            rho = self.zero_state()
        for g in circuit:
            rho = self._apply_op(rho, g.mat, g.qubits)
            for terms, qubits in noise_fn(g):
                rho = self.apply_terms(rho, terms, qubits)
        return rho

    def expect(self, rho, O):
        return float(np.real(np.trace(O @ rho)))


def pauli_terms(coeffs, k):
    ops = pauli_ops(k)
    return [(float(c), P) for c, P in zip(coeffs, ops)]


def kraus_terms(kraus_list):
    return [(1.0, K) for K in kraus_list]
