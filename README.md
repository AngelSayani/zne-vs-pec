# Choosing between zero-noise extrapolation and probabilistic error cancellation at a fixed shot budget

Code and data for:

> A. Sayani, *Choosing between zero-noise extrapolation and probabilistic error cancellation at a fixed shot budget: closed-form thresholds, a light-cone budget rule, and validation on a superconducting processor* (2026).

Everything in the paper is reproduced by the scripts in this repository. There are no
quantum-software dependencies; the simulator is written from scratch in NumPy.

## Requirements

Python 3.10 or later with `numpy`, `scipy` and `matplotlib`:

    pip install numpy scipy matplotlib

## Reproducing the paper

    python test_sim.py            # correctness checks of the simulator (Appendix A)
    python run_experiments.py     # exact noisy expectation values, 162 configurations -> results/exact.json
    python analysis.py            # Figures 1-5 and results/summary.json
    python run_ampdamp.py         # amplitude-damping study -> results/ampdamp.json
    python analysis_ampdamp.py    # Figure 6 and results/ampdamp_summary.json
    python run_lightcone.py       # light-cone PEC overheads -> results/lightcone.json, Figure 7
    python hardware/hw_analysis.py  # analyses the recorded device data -> Figure 8, results/hardware_summary.json

Total run time of the simulations is a few minutes on two CPU cores. The hardware analysis works from the recorded counts and needs no device access; re-running the device jobs used 470 s of QPU time on the IBM open plan. Figures are written to `figures/`.

## Files

| File | Purpose |
|---|---|
| `qsim.py` | Density-matrix simulator with depolarizing noise, global folding, batched Pauli-trajectory sampler |
| `noise_general.py` | General Pauli-channel inversion, amplitude damping, Kraus-sum density-matrix evolution |
| `circuits.py` | TFIM Trotter, hardware-efficient ansatz and GHZ circuit families |
| `estimators.py` | ZNE extrapolators, delta-method variance, PEC cost model, shot-level Monte Carlo |
| `run_experiments.py`, `analysis.py` | Main study (Sections 4-5) |
| `run_ampdamp.py`, `analysis_ampdamp.py` | Amplitude-damping robustness study |
| `run_lightcone.py` | Light-cone PEC overheads and exactness check |
| `hardware/` | Scripts used to run the IBM Quantum jobs (`pilot.py`, `run_main.py`, `run_pec.py`) and the analysis (`hw_analysis.py`); requires `qiskit`, `qiskit-ibm-runtime` and an IBM Quantum account for re-running |
| `results/hardware/` | Raw device counts, job identifiers and metadata for every hardware job (ibm_fez, 24 October 2026) |
| `results/` | All raw numbers used in the paper |

## License

MIT License. See `LICENSE`.
