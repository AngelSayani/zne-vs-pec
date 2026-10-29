"""Shared helpers for the hardware runs (IBM Quantum open plan)."""
import json, sys, time
import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService

import os
CREDS = os.environ.get("IBM_QUANTUM_CREDS", os.path.expanduser("~/.ibm_quantum_creds.json"))  # JSON with "token" and "instance"


def service():
    c = json.load(open(CREDS))
    return QiskitRuntimeService(channel="ibm_quantum_platform", token=c["token"], instance=c["instance"])


def tfim_qc(n, steps, J=1.0, h=1.0, dt=0.1):
    """Same circuit as circuits.tfim_trotter: RZZ(-2J dt) even then odd bonds, RX(-2h dt) all."""
    qc = QuantumCircuit(n)
    for _ in range(steps):
        for start in (0, 1):
            for i in range(start, n - 1, 2):
                qc.rzz(-2 * J * dt, i, i + 1)
        for i in range(n):
            qc.rx(-2 * h * dt, i)
    return qc


def ghz_qc(n):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.h(i + 1); qc.cz(i, i + 1); qc.h(i + 1)
    return qc


def fold(isa_circ, scale):
    """Global folding U (U^dag U)^k on an ISA circuit (no measurements)."""
    k = (scale - 1) // 2
    out = isa_circ.copy()
    inv = isa_circ.inverse()
    for _ in range(k):
        out = out.compose(inv).compose(isa_circ)
    return out


def best_linear_chain(backend, n):
    """Pick n physical qubits forming a path with the lowest summed two-qubit error."""
    props = backend.properties()
    cmap = backend.coupling_map
    edges = set()
    for a, b in cmap.get_edges():
        edges.add((min(a, b), max(a, b)))
    err = {}
    for (a, b) in edges:
        try:
            err[(a, b)] = props.gate_error("cz", [a, b])
        except Exception:
            try:
                err[(a, b)] = props.gate_error("cz", [b, a])
            except Exception:
                err[(a, b)] = 1.0
    adj = {}
    for (a, b) in edges:
        adj.setdefault(a, []).append(b); adj.setdefault(b, []).append(a)
    best = (1e9, None)
    def ro(q):
        try: return props.readout_error(q)
        except Exception: return 0.05
    def dfs(path, cost):
        nonlocal best
        if len(path) == n:
            c = cost + sum(ro(q) for q in path)
            if c < best[0]: best = (c, list(path))
            return
        for nb in adj.get(path[-1], []):
            if nb not in path:
                e = err.get((min(path[-1], nb), max(path[-1], nb)), 1.0)
                if cost + e < best[0]:
                    dfs(path + [nb], cost + e)
    for q in adj:
        dfs([q], 0.0)
    return best[1], best[0]


def z_expectations(counts, n):
    """Return <Z_i> for each qubit and the full bitstring distribution from a counts dict."""
    tot = sum(counts.values())
    z = np.zeros(n)
    for bs, c in counts.items():
        bits = bs.replace(" ", "")[::-1]  # qiskit little-endian -> index by qubit
        for i in range(n):
            z[i] += c * (1 - 2 * int(bits[i]))
    return z / tot, tot
