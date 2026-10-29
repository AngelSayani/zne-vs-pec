"""
Minimal exact simulator for noisy circuits with local depolarizing noise.

Two back-ends:
  * density-matrix (exact expectation values, arbitrary linear depolarizing
    maps including non-physical lambda > 1 used for miscalibrated PEC);
  * batched statevector Pauli-trajectory sampler (shot-level Monte Carlo
    for randomized PEC and ZNE validation).

Conventions: qubit 0 is the most significant axis.  A circuit is a list of
Gate(name, qubits, matrix).  Noise: after every gate a depolarizing channel
with parameter lambda acts on the gate's qubits:
    D_lambda(rho) = lambda * rho + (1 - lambda) * Tr_q(rho) (x) I/2^k
For physical depolarizing with error probability p (uniform over the
non-identity Paulis): lambda1 = 1 - 4p/3, lambda2 = 1 - 16p/15.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Sequence

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
PAULIS1 = [I2, X, Y, Z]


@dataclass
class Gate:
    name: str
    qubits: tuple
    mat: np.ndarray  # 2^k x 2^k unitary

    def dagger(self):
        return Gate(self.name + "^dag", self.qubits, self.mat.conj().T)


def rx(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta):
    return np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)]).astype(complex)


def rzz(theta):
    # exp(-i theta/2 Z Z)
    d = np.exp(-1j * theta / 2 * np.array([1, -1, -1, 1]))
    return np.diag(d).astype(complex)


CZ = np.diag([1, 1, 1, -1]).astype(complex)
CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


# ---------------------------------------------------------------- lambda / gamma
def lam_from_p(p: float, k: int) -> float:
    d = 4 ** k
    return 1.0 - p * d / (d - 1)


def gamma_from_lam(lam: float, k: int) -> float:
    """Sampling overhead (1-norm of quasi-probability) of the inverse of a
    k-qubit depolarizing channel with parameter lam."""
    d = 4 ** k
    # inverse channel has lambda' = 1/lam; Pauli coefficients:
    a0 = 1 / lam + (1 - 1 / lam) / d
    aP = (1 - 1 / lam) / d  # each of the d-1 non-identity Paulis
    return abs(a0) + (d - 1) * abs(aP)


def inverse_quasiprob(lam: float, k: int):
    """Quasi-probability coefficients (identity first) of inverse depolarizing."""
    d = 4 ** k
    a0 = 1 / lam + (1 - 1 / lam) / d
    aP = (1 - 1 / lam) / d
    return np.array([a0] + [aP] * (d - 1))


# ---------------------------------------------------------------- density matrix
class DensityMatrixSim:
    def __init__(self, n: int):
        self.n = n
        self.dim = 2 ** n

    def zero_state(self):
        rho = np.zeros((self.dim, self.dim), dtype=complex)
        rho[0, 0] = 1.0
        return rho

    def _apply_unitary(self, rho, U, qubits):
        n = self.n
        k = len(qubits)
        t = rho.reshape([2] * (2 * n))
        Ut = U.reshape([2] * (2 * k))
        # left multiply: contract U's input indices with rho's row axes
        t = np.tensordot(Ut, t, axes=(list(range(k, 2 * k)), list(qubits)))
        # result axes: U out (k) + remaining; move them back into place
        t = np.moveaxis(t, list(range(k)), list(qubits))
        # right multiply by U^dagger: contract conj(U) input indices with column axes
        col = [n + q for q in qubits]
        Uc = Ut.conj()
        t = np.tensordot(t, Uc, axes=(col, list(range(k, 2 * k))))
        # new axes appended at end correspond to U^* out indices
        t = np.moveaxis(t, list(range(2 * n - k, 2 * n)), col)
        return t.reshape(self.dim, self.dim)

    def _depolarize(self, rho, lam, qubits):
        if lam == 1.0:
            return rho
        return lam * rho + (1 - lam) * self._tensor_identity(rho, qubits)

    def _tensor_identity(self, rho, qubits):
        """Return Tr_q(rho) (x) I/2^k embedded on the full space."""
        n = self.n
        k = len(qubits)
        t = rho.reshape([2] * (2 * n))
        idx_row = list(range(n))
        idx_col = list(range(n, 2 * n))
        # einsum spec: trace over qubits (same letter for row and col)
        letters = "abcdefghijklmnopqrstuvwxyz"
        row_l = [letters[i] for i in range(n)]
        col_l = [letters[n + i] for i in range(n)]
        for q in qubits:
            col_l[q] = row_l[q]
        out_row = [row_l[i] for i in range(n) if i not in qubits]
        out_col = [col_l[i] for i in range(n) if i not in qubits]
        spec = "".join(row_l + col_l) + "->" + "".join(out_row + out_col)
        red = np.einsum(spec, t)  # shape 2^(n-k) x 2^(n-k) as tensor
        # re-embed with identity on traced qubits
        full = np.zeros([2] * (2 * n), dtype=complex)
        # build by iterating over basis of traced qubits
        import itertools
        red_t = red.reshape([2] * (2 * (n - k)))
        keep = [i for i in range(n) if i not in qubits]
        for bits in itertools.product([0, 1], repeat=k):
            sl_row = [slice(None)] * n
            sl_col = [slice(None)] * n
            for q, b in zip(qubits, bits):
                sl_row[q] = b
                sl_col[q] = b
            full[tuple(sl_row + sl_col)] += red_t / (2 ** k)
        return full.reshape(self.dim, self.dim)

    def run(self, circuit: Sequence[Gate], lam1: float, lam2: float, rho=None):
        if rho is None:
            rho = self.zero_state()
        for g in circuit:
            rho = self._apply_unitary(rho, g.mat, g.qubits)
            lam = lam1 if len(g.qubits) == 1 else lam2
            rho = self._depolarize(rho, lam, g.qubits)
        return rho

    def expect(self, rho, O):
        return float(np.real(np.trace(O @ rho)))


def pauli_string(n: int, ops: dict) -> np.ndarray:
    """Full 2^n matrix of a Pauli string, ops = {qubit: 'X'|'Y'|'Z'}."""
    m = {"I": I2, "X": X, "Y": Y, "Z": Z}
    out = np.array([[1.0 + 0j]])
    for q in range(n):
        out = np.kron(out, m[ops.get(q, "I")])
    return out


# ---------------------------------------------------------------- folding
def fold_global(circuit: Sequence[Gate], scale: int) -> List[Gate]:
    """Global unitary folding U -> U (U^dag U)^k with scale = 2k+1."""
    assert scale % 2 == 1
    k = (scale - 1) // 2
    out = list(circuit)
    inv = [g.dagger() for g in reversed(circuit)]
    for _ in range(k):
        out = out + inv + list(circuit)
    return out


def gate_counts(circuit):
    n1 = sum(1 for g in circuit if len(g.qubits) == 1)
    n2 = sum(1 for g in circuit if len(g.qubits) == 2)
    return n1, n2


# ---------------------------------------------------------------- trajectories
class TrajectorySim:
    """Batched statevector simulation with per-trajectory Pauli insertions.

    Used for shot-level Monte Carlo of PEC (sampled inverse) and for
    validating the density-matrix results.  Noise is local depolarizing
    (a Pauli channel), so every trajectory is pure.
    """

    def __init__(self, n: int, rng: np.random.Generator):
        self.n = n
        self.rng = rng

    def _apply_u(self, psi, U, qubits):
        # psi shape (B, 2,...,2)
        n = self.n
        k = len(qubits)
        Ut = U.reshape([2] * (2 * k))
        axes = [1 + q for q in qubits]
        t = np.tensordot(psi, Ut, axes=(axes, list(range(k, 2 * k))))
        # new axes appended at end -> move to qubit positions
        t = np.moveaxis(t, list(range(t.ndim - k, t.ndim)), axes)
        return t

    def _apply_pauli_batch(self, psi, idx, qubit):
        """idx: array (B,) of Pauli indices 0..3 to apply on `qubit` per trajectory."""
        n = self.n
        ax = 1 + qubit
        # X component (idx 1 or 2): flip
        flip = (idx == 1) | (idx == 2)
        if flip.any():
            flipped = np.flip(psi, axis=ax)
            psi = np.where(flip.reshape([-1] + [1] * n), flipped, psi)
        # Z component (idx 2 or 3): sign on |1>
        zsign = (idx == 2) | (idx == 3)
        if zsign.any():
            shape = [1] * (n + 1)
            shape[ax] = 2
            sgn = np.array([1, -1], dtype=complex).reshape(shape)
            signed = psi * sgn
            psi = np.where(zsign.reshape([-1] + [1] * n), signed, psi)
        # Y has extra phase i for X then Z ordering: Y = i X Z ; global phase per
        # trajectory is irrelevant for expectation values -> ignored.
        return psi

    def sample_pauli(self, B, k, probs):
        """Sample Pauli indices (B, k) from k-qubit Pauli distribution `probs`
        over 4^k strings (identity first)."""
        flat = self.rng.choice(4 ** k, size=B, p=probs)
        out = np.zeros((B, k), dtype=int)
        for j in range(k):
            out[:, k - 1 - j] = (flat // (4 ** j)) % 4
        return out

    def run(self, circuit, B, noise_probs: dict, pec_quasi: dict | None = None):
        """Simulate B trajectories.

        noise_probs[k] : length-4^k probability vector of the physical Pauli
                         channel on k qubits (identity first).
        pec_quasi[k]   : quasi-probability coefficients of the inverse channel
                         (identity first) or None for no PEC.
        Returns (psi_batch, sign_batch) where sign_batch holds the product of
        quasi-probability signs for each trajectory.
        """
        n = self.n
        psi = np.zeros((B,) + (2,) * n, dtype=complex)
        psi[(slice(None),) + (0,) * n] = 1.0
        sign = np.ones(B)
        for g in circuit:
            k = len(g.qubits)
            psi = self._apply_u(psi, g.mat, g.qubits)
            # physical noise
            pidx = self.sample_pauli(B, k, noise_probs[k])
            for j, q in enumerate(g.qubits):
                psi = self._apply_pauli_batch(psi, pidx[:, j], q)
            if pec_quasi is not None:
                qp = pec_quasi[k]
                probs = np.abs(qp) / np.abs(qp).sum()
                flat = self.rng.choice(4 ** k, size=B, p=probs)
                sign = sign * np.sign(qp[flat])
                qidx = np.zeros((B, k), dtype=int)
                for j in range(k):
                    qidx[:, k - 1 - j] = (flat // (4 ** j)) % 4
                for j, q in enumerate(g.qubits):
                    psi = self._apply_pauli_batch(psi, qidx[:, j], q)
        return psi, sign

    def measure_pauli_z_string(self, psi, qubits):
        """Return one sampled eigenvalue (+-1) per trajectory of a Z-type
        Pauli string on `qubits` (apply basis-change gates beforehand)."""
        B = psi.shape[0]
        probs = np.abs(psi.reshape(B, -1)) ** 2
        probs = probs / probs.sum(axis=1, keepdims=True)
        # sample bitstrings
        cum = np.cumsum(probs, axis=1)
        r = self.rng.random((B, 1))
        idx = (r > cum).sum(axis=1)
        idx = np.minimum(idx, probs.shape[1] - 1)
        val = np.ones(B)
        for q in qubits:
            bit = (idx >> (self.n - 1 - q)) & 1
            val = val * (1 - 2 * bit)
        return val


def depolarizing_probs(p: float, k: int):
    d = 4 ** k
    v = np.full(d, p / (d - 1))
    v[0] = 1 - p
    return v
