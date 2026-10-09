# Zero-noise extrapolation versus probabilistic error cancellation on noisy quantum processors

Code, data and manuscript for:

> A. Sayani, *Zero-noise extrapolation versus probabilistic error cancellation on noisy quantum processors: a fixed-budget bias–variance analysis and selection rule* (2026).

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

Total run time is a few minutes on two CPU cores. Figures are written to `figures/`.
The manuscript source is in `paper/` (compile with `latexmk -pdf main.tex`).

## Files

| File | Purpose |
|---|---|
| `qsim.py` | Density-matrix simulator with depolarizing noise, global folding, batched Pauli-trajectory sampler |
| `noise_general.py` | General Pauli-channel inversion, amplitude damping, Kraus-sum density-matrix evolution |
| `circuits.py` | TFIM Trotter, hardware-efficient ansatz and GHZ circuit families |
| `estimators.py` | ZNE extrapolators, delta-method variance, PEC cost model, shot-level Monte Carlo |
| `run_experiments.py`, `analysis.py` | Main study (Sections 4-5) |
| `run_ampdamp.py`, `analysis_ampdamp.py` | Amplitude-damping robustness study (Section 4.5) |
| `results/` | All raw numbers used in the paper |

## License

MIT License. See `LICENSE`.
