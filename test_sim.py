import numpy as np, time
from qsim import *
from circuits import *

rng = np.random.default_rng(1)

# 1. Noiseless density matrix vs trajectory for TFIM
n = 4
circ = tfim_trotter(n, 3)
O, meas_q, bc = tfim_observable(n)
dm = DensityMatrixSim(n)
rho = dm.run(circ, 1.0, 1.0)
e_ideal = dm.expect(rho, O)
ts = TrajectorySim(n, rng)
psi, s = ts.run(circ, 2000, {1: depolarizing_probs(0, 1), 2: depolarizing_probs(0, 2)})
amp = psi.reshape(2000, -1)
rho_t = np.einsum("bi,bj->ij", amp, amp.conj()) / 2000
print("ideal dm", e_ideal, "traj", np.real(np.trace(O @ rho_t)))

# 2. Noisy: dm with lambda vs trajectory average
p1, p2 = 0.01, 0.05
l1, l2 = lam_from_p(p1, 1), lam_from_p(p2, 2)
rho = dm.run(circ, l1, l2)
e_noisy = dm.expect(rho, O)
B = 20000
psi, s = ts.run(circ, B, {1: depolarizing_probs(p1, 1), 2: depolarizing_probs(p2, 2)})
amp = psi.reshape(B, -1)
ev = np.real(np.einsum("bi,ij,bj->b", amp.conj(), O, amp))
print("noisy dm", e_noisy, "traj", ev.mean(), "+-", ev.std() / np.sqrt(B))

# 3. PEC trajectories: unbiased recovery of ideal value
qp = {1: inverse_quasiprob(l1, 1), 2: inverse_quasiprob(l2, 2)}
n1, n2 = gate_counts(circ)
gam = gamma_from_lam(l1, 1) ** n1 * gamma_from_lam(l2, 2) ** n2
psi, s = ts.run(circ, B, {1: depolarizing_probs(p1, 1), 2: depolarizing_probs(p2, 2)}, pec_quasi=qp)
amp = psi.reshape(B, -1)
ev = np.real(np.einsum("bi,ij,bj->b", amp.conj(), O, amp))
est = gam * s * ev
print("PEC est", est.mean(), "+-", est.std() / np.sqrt(B), "gamma", gam, "ideal", e_ideal)
# single-shot PEC
shots = ts.measure_pauli_z_string(psi, meas_q)
est1 = gam * s * shots
print("PEC single-shot", est1.mean(), "+-", est1.std() / np.sqrt(B),
      "var", est1.var(), "pred gamma^2-O^2", gam ** 2 - e_ideal ** 2)

# 4. Miscalibrated PEC via lambda ratio in dm
delta = 0.2
l1h, l2h = lam_from_p(p1 * (1 + delta), 1), lam_from_p(p2 * (1 + delta), 2)
rho_res = dm.run(circ, l1 / l1h, l2 / l2h)
print("miscal PEC (dm residual)", dm.expect(rho_res, O))
qph = {1: inverse_quasiprob(l1h, 1), 2: inverse_quasiprob(l2h, 2)}
gamh = gamma_from_lam(l1h, 1) ** n1 * gamma_from_lam(l2h, 2) ** n2
psi, s = ts.run(circ, B, {1: depolarizing_probs(p1, 1), 2: depolarizing_probs(p2, 2)}, pec_quasi=qph)
amp = psi.reshape(B, -1)
ev = np.real(np.einsum("bi,ij,bj->b", amp.conj(), O, amp))
print("miscal PEC traj", (gamh * s * ev).mean(), "+-", (gamh * s * ev).std() / np.sqrt(B))

# 5. GHZ parity
n = 5
circ = ghz(n)
O, mq, bc = ghz_observable(n)
rho = dm.__class__(n).run(circ, 1.0, 1.0)
print("GHZ ideal parity", DensityMatrixSim(n).expect(rho, O))
t0 = time.time()
rho = DensityMatrixSim(8).run(ghz(8), l1, l2)
print("n=8 dm time", time.time() - t0)
t0 = time.time()
rho = DensityMatrixSim(6).run(tfim_trotter(6, 10), l1, l2)
print("n=6 tfim 10 steps dm time", time.time() - t0, gate_counts(tfim_trotter(6, 10)))
