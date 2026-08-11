# sphalerite-lattice-mc

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21812811.svg)](https://doi.org/10.5281/zenodo.21812811)

Statistical thermodynamic lattice model for Cu–In coupled substitution in
sphalerite. Four-term configurational Hamiltonian with a parameter-free
charge-compensation term, Metropolis Monte Carlo sampling, and
machine-precision analytic validation.

This repository contains the complete source code, numerical results and
manuscript for:

> **He Yu** and **Ruqing Chen**, *A Statistical Thermodynamic Lattice Model for
> Cu–In Coupled Substitution in Sphalerite.*

- He Yu — Geological Laboratory, Hezhou University, Hezhou, Guangxi 542899, China (`yuhe@hzxy.edu.cn`)
- Ruqing Chen — GUT Geoservice Inc., Montreal, Quebec, Canada (`ruqing@hotmail.com`)
- Huanzhang Lu — Département des sciences appliquées et Centre d'études sur les ressources minérales, Université du Québec à Chicoutimi, Quebec, Canada (`hzlu@uqac.uquebec.ca`)

---

## What the model does

Indium in sphalerite is almost invariably accompanied by copper in a near-1:1
molar ratio, universally attributed to the coupled substitution
2 Zn²⁺ ↔ Cu⁺ + In³⁺. That relation has been documented extensively but never
derived. This code derives it.

The total configurational energy is

```
H = E_c + E_s + E_sub + E_def
```

where `E_c` penalises departures from local electroneutrality on the sulphur
coordination tetrahedra, `E_s` is the elastic misfit energy, `E_sub` the
substitution energy referenced to the ore fluid, and `E_def` the cation-vacancy
term.

The central technical device is an exact topological property of the zinc-blende
structure: **two nearest-neighbour cations share precisely one bridging anion.**
This lets the tetrahedral electroneutrality functional be reduced *without
approximation* to a composition constant plus a nearest-neighbour pair
interaction,

```
E_c = 4λ Σ_i δq_i²  +  2λ Σ_<ij> δq_i δq_j
```

whose coefficient is then fixed by the dielectric constant of ZnS rather than
fitted: λ = e²/(8πε₀ε_r d_NN) = 0.227–0.368 eV.

## Headline results

| Quantity | Value |
|---|---|
| Cu–In pair binding energy | −2λ ≈ −0.45 to −0.73 eV (no fitted parameter) |
| Ground state of `E_c` at Cu:In = 1:1 | exactly the roquesite (CuInS₂) cation ordering, `E_c` = 0 |
| Cu–In bond enrichment at 300 °C, electrostatics only | 16.0 × random |
| Cu–In bond enrichment at 300 °C, full model | 24.5 × random |
| Local charge imbalance removed | 96 % (electrostatics only) |
| Heterovalent / homovalent enrichment ratio | 2.70 |
| Ordering crossover | Θ\* ≈ 0.6–0.8, i.e. 3–7× above the entire hydrothermal window |
| Elastic contribution | undetectable; largest strain pair term is 1.1 meV, ~40× below *k*<sub>B</sub>*T* |
| Cu route vs vacancy route | Cu favoured by 8λ ≈ 2.4 eV |

## Repository layout

```
src/
  sphalerite_mc.py            core module: lattice, Hamiltonian, MC engine,
                              order parameters, verification suite
  run_experiments.py          Experiments A–D; regenerates all figures and data
  analytic_ground_states.py   annealed minimisation of E_c; ground states and
                              compensation-route comparison
data/
  results.csv                 numerical results underlying Figs. 1–4
  run_log.txt                 full console output of the production run
figures/
  fig1_annealing.{pdf,png}    annealing scan, three model variants
  fig2_lambda_scan.{pdf,png}  sensitivity to λ, three seeds per value
  fig3_cluster.{pdf,png}      cluster-size distribution and slab projections
  fig4_ergodicity.{pdf,png}   quenched vs pre-annealed trajectories
paper/
  sphalerite_cu_in.tex        manuscript source (English)
  sphalerite_cu_in.pdf        compiled manuscript (English)
  sphalerite_cu_in_zh.tex     manuscript source (Chinese translation)
  sphalerite_cu_in_zh.pdf     compiled manuscript (Chinese translation)
```

## Requirements

```
python >= 3.10
numpy
matplotlib      # figures only
numba           # optional; ~50× speed-up, strongly recommended
```

```bash
pip install numpy matplotlib numba
```

## Reproducing everything

```bash
cd src

# 1. Verification and validation suite (~5 s).
#    Checks lattice topology, energy bookkeeping, every analytic limit,
#    and the random-solution limit of the order parameters.
python sphalerite_mc.py

# 2. Analytic ground states (~1 min).
#    Reproduces the isolated-defect costs, the -2λ pair binding energy,
#    the E_c = 0 roquesite ground state and its degeneracy, and the
#    8λ preference for the Cu-compensated route.
python analytic_ground_states.py

# 3. Full experiment suite (~8 min single-threaded with Numba).
#    Writes results_phase1.csv and Figs. 1-4 as both PDF and PNG.
python run_experiments.py
```

All runs use fixed seeds and are bitwise reproducible on a given platform. The
published figures were produced in 507 s on a single core.

Building the manuscript:

```bash
cd paper
pdflatex sphalerite_cu_in.tex      # English; run twice for cross-references
xelatex  sphalerite_cu_in_zh.tex   # Chinese; requires ctex + Noto CJK fonts
```

The Chinese version is a translation of the English manuscript and is
content-identical to it. Figure axis labels remain in English; figure captions
are translated. On Debian/Ubuntu the Chinese build needs
`texlive-lang-chinese` and `fonts-noto-cjk`.

## What this model cannot do

We state the limitations plainly because they define the scope of the
conclusions.

- **Finite size.** At *N* = 4000 sites and 4 at.% total solute, the 160 solute
  atoms are exactly enough to form one condensed domain. Once nucleation
  occurs the entire solute budget is consumed, so the model **cannot resolve
  the solid-solution versus nanoinclusion boundary**. The saturation of Fig. 2
  above λ = 0.10 eV and the large seed-to-seed dispersion above 0.15 eV are
  finite-size artefacts and should not be interpreted physically. Settling the
  question needs ~50³ cells, enhanced sampling, and a composition scan.
- **Acceptance collapse.** Below 1300 K the acceptance ratio falls to ~10⁻⁵.
  Experiment D shows the low-temperature plateau is nonetheless thermodynamic
  rather than an artefact, but simple Kawasaki dynamics is at its limit; larger
  studies will need parallel tempering or cluster moves.
- **Unconstrained parameters.** `J`, `Γ₁`, `ε_□` and `E_V^f` are hypotheses
  awaiting first-principles determination. All principal results are reported
  both with and without `J`.
- **No Fe.** Natural sphalerite carries up to 10 mol% FeS, which will modify
  both `ε_r` and the substitution energies.
- **Monte Carlo sweeps are not physical time.** The model makes no kinetic
  predictions.

## Notes for anyone building on this

Two bugs were found and fixed during development, both of which changed results
qualitatively. They are documented in the source comments and each is guarded by
a regression assertion. If you adapt this code, keep those assertions.

1. **Solute index list corruption.** When both exchanged sites are solutes
   (a Cu ↔ In swap) the list must be left untouched; replacing the entry
   unconditionally enters one site twice and drops the other, silently
   corrupting the proposal distribution over long runs.
2. **Pair-count double counting.** In the bond matrix, off-diagonal (heterovalent)
   entries must **not** be halved — each such bond is counted once, from one end
   only. Halving them understates the Cu–In bond count and α by a factor of two.

The identity `R_ab = 1 − α_ba` and the random-solution limit `R → 1` are both
asserted in the test suite and catch either error immediately.

## Citation

Yu, H., Chen, R., Lu, H. (2026) *A Statistical Thermodynamic Lattice Model for
Cu–In Coupled Substitution in Sphalerite.* Zenodo.
https://doi.org/10.5281/zenodo.21812811

See `CITATION.cff` for machine-readable metadata.

## Funding

Hezhou Municipal Scientific Research and Development Program, Project Nos.
2024143, 2024141 and 2024104.

## Licence

- **Code** (`src/`): MIT — see `LICENSE`.
- **Data and figures** (`data/`, `figures/`): CC BY 4.0 — see `LICENSE-DATA`.
