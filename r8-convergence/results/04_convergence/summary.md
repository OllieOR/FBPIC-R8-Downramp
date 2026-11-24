# R8 convergence comparison

Four completed R8 runs, with only azimuthal particle sampling and retained modes changed. The spatial grid, timestep, density profile, input laser and seed are checked for equality.

The main accelerated population uses forward electrons in the first complete wake period and K ≥20 MeV. Near-axis charge applies r ≤2 µm. Moments require at least 20 effective macroparticles. The particle selection is evaluated independently at each saved iteration.

NRMSE uses time-weighted squared differences over adjacent valid diagnostics. The percentage is normalised by the reference signal RMS. Missing windows and insufficient particle counts remain undefined.

## Particle sampling: p_nt 8 → 16, Nm=2

| Quantity | Time-weighted NRMSE | Final reference | Final comparison | Final difference | Valid time |
|---|---:|---:|---:|---:|---:|
| Main accelerated charge ≥20 MeV (pC) | 0.577% | 25.6 | 25.6 | -0.0736% | 87% |
| Near-axis main accelerated charge ≥20 MeV (pC) | 1.28% | 0.204 | 0.126 | — | 87% |
| Main population mean energy (MeV) | 0.356% | 30.2 | 30.2 | 0.0362% | 71% |
| Main population relative RMS energy spread (fraction) | 2.13% | 0.226 | 0.224 | -0.873% | 71% |
| Normalised emittance x (mm mrad) | 3.42% | 17.5 | 16.8 | -3.88% | 71% |
| Normalised emittance y (mm mrad) | 5.94% | 3.31 | 3.49 | 5.43% | 71% |
| RMS divergence x (mrad) | 1.22% | 82.8 | 83 | 0.312% | 71% |
| RMS divergence y (mrad) | 4.88% | 38 | 38.9 | 2.32% | 71% |
| Maximum forward energy (MeV) | 2.71% | 62.4 | 63.9 | 2.39% | 97.7% |
| 99th percentile forward energy (MeV) | 0.138% | 43.1 | 43.2 | 0.236% | 97.7% |
| Saved optical energy (mJ) | 7.25e-05% | 262 | 262 | -6.6e-05% | 100% |
| Laser spot radius (µm) | 0.000691% | 21 | 21 | 5.43e-05% | 100% |
| Peak accelerating wake field (GV/m) | 0.102% | 56.9 | 56.6 | -0.41% | 100% |
| First complete wake period length (µm) | 0.13% | 19.5 | 19.5 | 0.0596% | 87% |
| Main accelerated charge ≥5 MeV (pC) | 0.31% | 42.3 | 42.3 | -0.198% | 87% |

## Azimuthal modes: Nm 2 → 3, p_nt=16

| Quantity | Time-weighted NRMSE | Final reference | Final comparison | Final difference | Valid time |
|---|---:|---:|---:|---:|---:|
| Main accelerated charge ≥20 MeV (pC) | 13.1% | 25.6 | 21.9 | -14.4% | 87% |
| Near-axis main accelerated charge ≥20 MeV (pC) | 12.4% | 0.126 | 0.118 | — | 87% |
| Main population mean energy (MeV) | 1.89% | 30.2 | 29.7 | -1.57% | 70.7% |
| Main population relative RMS energy spread (fraction) | 9.71% | 0.224 | 0.208 | -7.31% | 70.7% |
| Normalised emittance x (mm mrad) | 24.7% | 16.8 | 17.1 | 1.56% | 70.7% |
| Normalised emittance y (mm mrad) | 47.5% | 3.49 | 1.84 | -47.3% | 70.7% |
| RMS divergence x (mrad) | 13.7% | 83 | 89.6 | 7.89% | 70.7% |
| RMS divergence y (mrad) | 38% | 38.9 | 30.1 | -22.6% | 70.7% |
| Maximum forward energy (MeV) | 5.25% | 63.9 | 60.5 | -5.33% | 97.7% |
| 99th percentile forward energy (MeV) | 5.99% | 43.2 | 40.1 | -7.06% | 97.7% |
| Saved optical energy (mJ) | 0.000994% | 262 | 262 | -0.000217% | 100% |
| Laser spot radius (µm) | 0.0392% | 21 | 21 | -0.00434% | 100% |
| Peak accelerating wake field (GV/m) | 1.94% | 56.6 | 56.6 | -0.019% | 100% |
| First complete wake period length (µm) | 0.831% | 19.5 | 19.5 | 0.416% | 87% |
| Main accelerated charge ≥5 MeV (pC) | 5.2% | 42.3 | 39.7 | -6.1% | 87% |

## Azimuthal modes: Nm 3 → 4, p_nt=16

| Quantity | Time-weighted NRMSE | Final reference | Final comparison | Final difference | Valid time |
|---|---:|---:|---:|---:|---:|
| Main accelerated charge ≥20 MeV (pC) | 0.824% | 21.9 | 22.1 | 0.669% | 87% |
| Near-axis main accelerated charge ≥20 MeV (pC) | 1.99% | 0.118 | 0.0862 | — | 87% |
| Main population mean energy (MeV) | 0.254% | 29.7 | 29.7 | -0.0133% | 71.6% |
| Main population relative RMS energy spread (fraction) | 1.84% | 0.208 | 0.208 | 0.268% | 71.6% |
| Normalised emittance x (mm mrad) | 2.74% | 17.1 | 16.9 | -1.11% | 71.6% |
| Normalised emittance y (mm mrad) | 9.49% | 1.84 | 1.74 | -5.44% | 71.6% |
| RMS divergence x (mrad) | 2.32% | 89.6 | 88.9 | -0.74% | 71.6% |
| RMS divergence y (mrad) | 2.78% | 30.1 | 29.7 | -1.39% | 71.6% |
| Maximum forward energy (MeV) | 3.66% | 60.5 | 58.1 | -3.98% | 97.7% |
| 99th percentile forward energy (MeV) | 0.285% | 40.1 | 40.4 | 0.808% | 97.7% |
| Saved optical energy (mJ) | 9.16e-05% | 262 | 262 | -0.000186% | 100% |
| Laser spot radius (µm) | 0.01% | 21 | 21 | -6.97e-05% | 100% |
| Peak accelerating wake field (GV/m) | 0.176% | 56.6 | 56.6 | -0.0095% | 100% |
| First complete wake period length (µm) | 0.179% | 19.5 | 19.5 | -0.0351% | 87% |
| Main accelerated charge ≥5 MeV (pC) | 0.497% | 39.7 | 39.8 | 0.309% | 87% |

## Input and diagnostic checks

| Setting | Unresolved wake periods | Low effective count | Optical boundary flags |
|---|---:|---:|---:|
| Nm=2, p_nt=8 | 27 | 171 | 0 |
| Nm=2, p_nt=16 | 27 | 150 | 0 |
| Nm=3, p_nt=16 | 27 | 181 | 0 |
| Nm=4, p_nt=16 | 28 | 179 | 0 |

Charge is undefined until a complete wake period is resolved. Resolved empty selections have zero charge and undefined moments. The optical boundary flags record when part of the pulse approaches the edge of the saved window.

## Comparing the final snapshot

At fixed pₙₜ = 16, the final charge above 20 MeV in the main accelerated population went from 25.60 pC with two modes to 21.91 pC with three, then to 22.06 pC with four. I used this comparison when choosing three modes for the main beam analysis. The final maximum forward kinetic energy changed by -2.41 MeV when adding the fourth mode.

The particle-sampling test was performed at Nₘ = 2, so the effect of increasing pₙₜ at Nₘ = 3 remains unknown. The spatial grid, timestep and seed were kept fixed throughout.
