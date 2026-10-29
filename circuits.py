"""Structured circuit families used in the study."""
import numpy as np
from qsim import Gate, rx, ry, rzz, CZ, H, pauli_string


def tfim_trotter(n: int, steps: int, J=1.0, h=1.0, dt=0.1):
    """First-order Trotterised transverse-field Ising dynamics on an open chain.
    H = -J sum Z_i Z_{i+1} - h sum X_i.  Returns gate list."""
    circ = []
    for _ in range(steps):
        # even bonds then odd bonds
        for start in (0, 1):
            for i in range(start, n - 1, 2):
                circ.append(Gate("RZZ", (i, i + 1), rzz(-2 * J * dt)))
        for i in range(n):
            circ.append(Gate("RX", (i,), rx(-2 * h * dt)))
    return circ


def tfim_observable(n: int):
    """Middle-site magnetisation Z_{n//2} (Z-basis measurement, no basis change)."""
    return pauli_string(n, {n // 2: "Z"}), [n // 2], []


def hea(n: int, layers: int, seed: int = 7):
    """Hardware-efficient ansatz: RY layer + CZ ladder, fixed random angles."""
    rng = np.random.default_rng(seed)
    circ = []
    for _ in range(layers):
        for i in range(n):
            circ.append(Gate("RY", (i,), ry(rng.uniform(0, 2 * np.pi))))
        for i in range(0, n - 1, 2):
            circ.append(Gate("CZ", (i, i + 1), CZ))
        for i in range(1, n - 1, 2):
            circ.append(Gate("CZ", (i, i + 1), CZ))
    for i in range(n):
        circ.append(Gate("RY", (i,), ry(rng.uniform(0, 2 * np.pi))))
    return circ


def hea_observable(n: int):
    return pauli_string(n, {0: "Z"}), [0], []


def ghz(n: int):
    """GHZ preparation: H on 0, CX ladder (CX = H CZ H built from CZ+H)."""
    circ = [Gate("H", (0,), H)]
    for i in range(n - 1):
        circ.append(Gate("H", (i + 1,), H))
        circ.append(Gate("CZ", (i, i + 1), CZ))
        circ.append(Gate("H", (i + 1,), H))
    return circ


def ghz_observable(n: int):
    """X^{(x)n} parity; measured by applying H to all qubits then Z-string."""
    basis_change = [Gate("H", (i,), H) for i in range(n)]
    return pauli_string(n, {i: "X" for i in range(n)}), list(range(n)), basis_change
