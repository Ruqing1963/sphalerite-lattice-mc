# -*- coding: utf-8 -*-
"""
================================================================================
 sphalerite_mc.py

 A statistical thermodynamic lattice model for Cu-In coupled substitution
 in sphalerite.

 Reference implementation accompanying:
   He Yu & Ruqing Chen, "A Statistical Thermodynamic Lattice Model for
   Cu-In Coupled Substitution in Sphalerite."

 Repository: https://github.com/Ruqing1963/sphalerite-lattice-mc
 Licence:    MIT (code)
 Requires:   numpy; numba (optional, for JIT acceleration)
--------------------------------------------------------------------------------
 MODEL
   Total configurational Hamiltonian:   H = E_c + E_s + E_sub + E_def

     E_c   Local charge-compensation penalty, defined on the sulphur
           coordination tetrahedra:  lambda * sum_a (Delta Q_a)^2
     E_s   Elastic misfit energy: Eshelby self-energy plus an elastic-dipole
           pair term decaying as r^-3
     E_sub Substitution energy: point-defect formation energies referenced to an
           external reservoir, plus a short-range chemical term J
     E_def Cation-vacancy term: formation energy plus association correction
--------------------------------------------------------------------------------
 LATTICE GEOMETRY (key implementation device)
   Sphalerite (ZnS, space group F-43m):
     - cations (Zn/Cu/In/vacancy) occupy an fcc sublattice, Z1 = 12, Z2 = 6
     - sulphur occupies a second fcc sublattice displaced by (1/4, 1/4, 1/4)
     - every S is tetrahedrally coordinated by four cations, and every cation
       by four S
     - any two nearest-neighbour cations share EXACTLY ONE bridging S. This
       topological property is what allows E_c to be reduced exactly to a
       nearest-neighbour pair interaction (see the Lemma in the paper).

   All coordinates are held as integers in units of a0/4 on a grid of edge
   G = 4L with periodic boundaries:
     cations: (4cx, 4cy, 4cz) + basis, basis in {(0,0,0),(0,2,2),(2,0,2),(2,2,0)}
     anions:  cation coordinate + (1,1,1)
   Adjacency is therefore pure integer table lookup, with no floating-point
   neighbour search, which keeps the kernel JIT-friendly and portable to much
   larger boxes.
================================================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

# ------------------------------------------------------------------------------
# 0. Optional Numba acceleration
# ------------------------------------------------------------------------------
try:
    from numba import njit
    HAS_NUMBA = True
except Exception:                                        # pragma: no cover
    HAS_NUMBA = False

    def njit(*args, **kwargs):                           # no-op fallback
        def _wrap(f):
            return f
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return _wrap


# ==============================================================================
# 1. Physical constants and species definitions
# ==============================================================================
KB_EV = 8.617333262e-5       # Boltzmann constant [eV/K]
E2_OVER_4PIEPS0 = 14.399645  # e^2 / (4*pi*eps0) [eV Angstrom]

# Species encoding: 0 = Zn(II), 1 = Cu(I), 2 = In(III), 3 = cation vacancy V_Zn
SPECIES_NAMES = ("Zn", "Cu", "In", "Vac")
N_SPECIES = 4
FORMAL_CHARGE = np.array([2.0, 1.0, 3.0, 0.0])   # formal charge q_alpha
DELTA_Q = FORMAL_CHARGE - 2.0                    # deviation from host Zn(II): 0, -1, +1, -2

# Shannon effective ionic radii, fourfold coordination [Angstrom]  -- measured data
SHANNON_IV = {"Zn": 0.60, "Cu": 0.60, "In": 0.62, "Ag": 1.00,
              "Fe": 0.63, "Cd": 0.78, "Mn": 0.66}


# ==============================================================================
# 2. Lattice construction
# ==============================================================================
class SphaleriteLattice:
    """
    Builds the sphalerite cation fcc sublattice, the sulphur sublattice, and all
    adjacency tables.

    Attributes
    ----------
    N          : number of cation sites (= number of anion sites) = 4 * L**3
    nn1        : (N, 12) int32  first cation neighbour shell, |r| = a0/sqrt(2)
    nn2        : (N,  6) int32  second cation neighbour shell, |r| = a0
    an_of_cat  : (N,  4) int32  the four S coordinating each cation
    cat_of_an  : (N,  4) int32  the four cations coordinating each S
    """

    # fcc basis, in units of a0/4
    _BASIS = np.array([[0, 0, 0], [0, 2, 2], [2, 0, 2], [2, 2, 0]], dtype=np.int64)

    def __init__(self, L: int, a0: float = 5.4093):
        self.L = int(L)
        self.a0 = float(a0)
        self.G = 4 * self.L                        # integer grid edge, units of a0/4
        self.N = 4 * self.L ** 3
        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        L, G = self.L, self.G

        # --- 2.1 cation integer coordinates ---------------------------
        cells = np.array([(cx, cy, cz)
                          for cz in range(L) for cy in range(L) for cx in range(L)],
                         dtype=np.int64)                       # (L^3, 3)
        pos = (cells[:, None, :] * 4 + self._BASIS[None, :, :]).reshape(-1, 3)
        self.pos = pos % G                                     # (N, 3)
        assert pos.shape[0] == self.N

        # --- 2.2 coordinate -> index lookup table ---------------------
        lut = -np.ones((G, G, G), dtype=np.int64)
        lut[self.pos[:, 0], self.pos[:, 1], self.pos[:, 2]] = np.arange(self.N)
        self._lut = lut

        # --- 2.3 first shell: the twelve permutations of (+-2, +-2, 0) -
        v1 = []
        for a in (2, -2):
            for b in (2, -2):
                v1 += [(a, b, 0), (a, 0, b), (0, a, b)]
        v1 = np.unique(np.array(v1, dtype=np.int64), axis=0)
        assert v1.shape[0] == 12, v1.shape
        self.nn1 = self._shift_lookup(v1).astype(np.int32)

        # --- 2.4 second shell: the six permutations of (+-4, 0, 0) -----
        v2 = np.array([(4, 0, 0), (-4, 0, 0), (0, 4, 0),
                       (0, -4, 0), (0, 0, 4), (0, 0, -4)], dtype=np.int64)
        self.nn2 = self._shift_lookup(v2).astype(np.int32)

        # --- 2.5 sulphur sublattice -----------------------------------
        # Convention: anion j sits at (cation j) + (1,1,1), giving a one-to-one
        # cation-anion indexing. The four cations coordinating anion j are then
        # j, j+(2,2,0), j+(2,0,2) and j+(0,2,2).
        off_cat_of_an = np.array([(0, 0, 0), (2, 2, 0), (2, 0, 2), (0, 2, 2)], dtype=np.int64)
        self.cat_of_an = self._shift_lookup(off_cat_of_an).astype(np.int32)
        # The four anions coordinating cation i are the inverse of the above.
        self.an_of_cat = self._shift_lookup(-off_cat_of_an).astype(np.int32)

        # --- 2.6 self-consistency checks ------------------------------
        self._validate()

    def _shift_lookup(self, vecs: np.ndarray) -> np.ndarray:
        """Apply integer offsets with periodic wrapping; return (N, len(vecs)) indices."""
        out = np.empty((self.N, len(vecs)), dtype=np.int64)
        for k, v in enumerate(vecs):
            p = (self.pos + v) % self.G
            out[:, k] = self._lut[p[:, 0], p[:, 1], p[:, 2]]
        assert out.min() >= 0, "adjacency lookup failed: offset does not land on a lattice site"
        return out

    def _validate(self):
        """Geometry checks, including the 'exactly one shared anion' Lemma."""
        i = 0
        for j in self.nn1[i]:
            assert i in self.nn1[j], "first-neighbour relation is not reciprocal"
        for a in self.an_of_cat[i]:
            assert i in self.cat_of_an[a], "cation-anion coordination is not reciprocal"
        # Lemma: nearest-neighbour cations share exactly one bridging anion.
        for j in self.nn1[i]:
            shared = np.intersect1d(self.an_of_cat[i], self.an_of_cat[j])
            assert shared.size == 1, f"NN pair shares {shared.size} anions (expected 1)"
        # Second neighbours share none.
        for j in self.nn2[i]:
            shared = np.intersect1d(self.an_of_cat[i], self.an_of_cat[j])
            assert shared.size == 0, "second neighbours should share no anion"


# ==============================================================================
# 3. Hamiltonian parameters
# ==============================================================================
@dataclass
class HamiltonianParams:
    """
    All adjustable parameters of the four-term Hamiltonian.
    Units: eV (energy), Angstrom (length), K (temperature).

    Basis tags used below
      [F] established fact  - measured crystallographic / elastic / dielectric data,
                              or an exact consequence of the lattice topology
      [I] inference         - derived from [F] inputs through a standard physical
                              framework, with no fitted constant
      [H] hypothesis        - an assumption of this work, to be constrained by
                              first-principles calculation
    """
    # ---- E_c : local charge compensation ---------------------------
    lam: float = 0.30          # [I] lambda = e^2 / (8 pi eps0 eps_r d_NN); eps_r = 5.13-8.3 -> 0.227-0.368 eV
    eps_r: float = 8.3         # [F] static dielectric constant of ZnS
    # ---- E_s : elastic misfit --------------------------------------
    # size misfit eps_alpha = (r_alpha - r_Zn) / r_Zn
    misfit: np.ndarray = field(default_factory=lambda: np.array([
        0.0,                                        # Zn   [F]
        (0.60 - 0.60) / 0.60,                       # Cu+  [F]  = 0.000
        (0.62 - 0.60) / 0.60,                       # In3+ [F]  = 0.0333
        -0.20,                                      # Vac  [H]  effective inward relaxation
    ]))
    bulk_modulus: float = 77.0   # [F] K(ZnS) ~ 77 GPa
    shear_modulus: float = 30.7  # [F] mu(ZnS) ~ 31 GPa (Voigt-Reuss-Hill)
    gamma1: float = 1.0          # [H] first-shell elastic dipole coupling [eV]
    # ---- E_sub : substitution --------------------------------------
    # Non-electrostatic short-range chemical term J[alpha, beta] (symmetric) [eV]
    J: np.ndarray = field(default_factory=lambda: np.array([
        # Zn     Cu     In     Vac
        [0.00,  0.00,  0.00,  0.00],   # Zn
        [0.00, +0.05, -0.10,  0.00],   # Cu   [H] Cu 3d - In 5s / S 3p hybridisation, -0.10
        [0.00, -0.10, +0.05,  0.00],   # In   [H] non-electrostatic like-solute repulsion, +0.05
        [0.00,  0.00,  0.00,  0.00],   # Vac
    ]))
    # Point-defect formation energies relative to the host, including the
    # reservoir reference [eV]
    E_form: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0]))
    # ---- E_def : vacancies -----------------------------------------
    E_vac: float = 2.0          # [H] V_Zn formation energy, placeholder

    # ---- derived quantities -----------------------------------------
    def eshelby_self_energy(self, a0: float = 5.4093) -> np.ndarray:
        """
        Eshelby elastic self-energy of a spherical misfitting inclusion:

            E_el = 24 pi mu_m K_p R^3 delta^2 / (3 K_p + 4 mu_m)

        R is the Wigner-Seitz radius of a cation site (Omega = a0^3 / 4).
        Returns the self-energy of each species [eV].
        """
        omega = a0 ** 3 / 4.0                      # volume per cation site [A^3]
        R = (3.0 * omega / (4.0 * np.pi)) ** (1 / 3)
        K, mu = self.bulk_modulus, self.shear_modulus
        pref_gpa_a3 = 24 * np.pi * mu * K * R ** 3 / (3 * K + 4 * mu)   # [GPa A^3]
        pref_ev = pref_gpa_a3 * 6.241509e-3                             # 1 GPa A^3 = 6.2415e-3 eV
        return pref_ev * self.misfit ** 2

    def lambda_from_dielectric(self, d_nn: float) -> float:
        """Parameter-free estimate of lambda from the screened Coulomb interaction."""
        return E2_OVER_4PIEPS0 / (2.0 * self.eps_r * d_nn)

    def build_tables(self, a0: float = 5.4093):
        """
        Pre-compute the lookup tables used by the Monte Carlo kernel:
          P1[s1, s2] first-shell pair energy  = J + gamma1 * eps_s1 * eps_s2
          P2[s1, s2] second-shell pair energy = gamma2 * eps_s1 * eps_s2,
                     with gamma2 = gamma1 * (d1/d2)^3 = gamma1 / 2^1.5
          SELF[s]    single-site energy = Eshelby self-energy + formation energy
        """
        eps = self.misfit
        P1 = self.J + self.gamma1 * np.outer(eps, eps)
        gamma2 = self.gamma1 / 2.0 ** 1.5          # elastic dipole decays as r^-3
        P2 = gamma2 * np.outer(eps, eps)
        SELF = self.eshelby_self_energy(a0) + self.E_form
        SELF = SELF.copy()
        SELF[3] += self.E_vac
        return (np.ascontiguousarray(P1, dtype=np.float64),
                np.ascontiguousarray(P2, dtype=np.float64),
                np.ascontiguousarray(SELF, dtype=np.float64))


# ==============================================================================
# 4. JIT-compiled kernels
# ==============================================================================
@njit(cache=True, fastmath=True)
def _anion_charges(spec, cat_of_an, dq):
    """Net local charge deviation Delta Q_a of each S coordination tetrahedron."""
    n = cat_of_an.shape[0]
    Q = np.zeros(n, dtype=np.float64)
    for a in range(n):
        s = 0.0
        for k in range(4):
            s += dq[spec[cat_of_an[a, k]]]
        Q[a] = s
    return Q


@njit(cache=True, fastmath=True)
def _total_energy(spec, nn1, nn2, Q, P1, P2, SELF, lam):
    """Total energy from scratch. Used for initialisation and for validation only;
    the Monte Carlo loop works with incremental dE."""
    E = 0.0
    for a in range(Q.shape[0]):
        E += lam * Q[a] * Q[a]                       # E_c
    for i in range(spec.shape[0]):
        si = spec[i]
        E += SELF[si]                                # E_sub self-term + E_def
        for k in range(nn1.shape[1]):
            E += 0.5 * P1[si, spec[nn1[i, k]]]       # 0.5 corrects double counting
        for k in range(nn2.shape[1]):
            E += 0.5 * P2[si, spec[nn2[i, k]]]
    return E


@njit(cache=True, fastmath=True)
def _mc_run(spec, Q, nn1, nn2, an_of_cat, dq, P1, P2, lam,
            beta, n_steps, solutes, seed):
    """
    Metropolis-Hastings sampling with Kawasaki (exchange) dynamics.

    - Canonical ensemble: two sites exchange species, so composition is conserved
      exactly. This is the appropriate constraint for a closed sphalerite grain
      equilibrating internally after growth.
    - Proposal: site i is drawn uniformly from the solute list, site j uniformly
      from all sites. The proposal is symmetric because the number of solutes is
      conserved and the two roles are interchangeable, so detailed balance holds.
    - In the dilute limit every trial move involves a solute, which accelerates
      sampling by roughly a factor 1/x relative to drawing both sites at random.

    Returns (cumulative energy change, acceptance ratio).
    """
    np.random.seed(seed)
    N = spec.shape[0]
    n_sol = solutes.shape[0]
    n_acc = 0
    dE_total = 0.0

    a_list = np.empty(8, dtype=np.int64)
    d_list = np.empty(8, dtype=np.float64)

    for _ in range(n_steps):
        i = solutes[np.random.randint(n_sol)]
        j = np.random.randint(N)
        si = spec[i]
        sj = spec[j]
        if si == sj:
            continue

        # ---------- (a) E_c increment: affected S tetrahedra ----------
        d = dq[sj] - dq[si]
        n_aff = 0
        for k in range(4):
            a_list[n_aff] = an_of_cat[i, k]
            d_list[n_aff] = d
            n_aff += 1
        for k in range(4):
            a = an_of_cat[j, k]
            found = False
            for m in range(n_aff):
                if a_list[m] == a:
                    d_list[m] -= d          # shared tetrahedron: changes cancel when i, j are neighbours
                    found = True
                    break
            if not found:
                a_list[n_aff] = a
                d_list[n_aff] = -d
                n_aff += 1

        dE = 0.0
        for m in range(n_aff):
            q0 = Q[a_list[m]]
            q1 = q0 + d_list[m]
            dE += lam * (q1 * q1 - q0 * q0)

        # ---------- (b) pair increments (E_s pair term + E_sub chemical term) ----
        for k in range(nn1.shape[1]):
            nb = nn1[i, k]
            if nb != j:
                s = spec[nb]
                dE += P1[sj, s] - P1[si, s]
        for k in range(nn1.shape[1]):
            nb = nn1[j, k]
            if nb != i:
                s = spec[nb]
                dE += P1[si, s] - P1[sj, s]
        for k in range(nn2.shape[1]):
            nb = nn2[i, k]
            if nb != j:
                s = spec[nb]
                dE += P2[sj, s] - P2[si, s]
        for k in range(nn2.shape[1]):
            nb = nn2[j, k]
            if nb != i:
                s = spec[nb]
                dE += P2[si, s] - P2[sj, s]
        # Note: an exchange leaves the sum of single-site self-energies unchanged,
        # so SELF makes no contribution to dE.

        # ---------- (c) Metropolis criterion ----------
        acc = False
        if dE <= 0.0:
            acc = True
        elif np.random.random() < np.exp(-beta * dE):
            acc = True

        if acc:
            spec[i] = sj
            spec[j] = si
            for m in range(n_aff):
                Q[a_list[m]] += d_list[m]
            # ---- maintain the solute index list ----
            # The solute migrates from i to j only when j was previously host Zn.
            # If i and j are both solutes (e.g. a Cu <-> In swap) both remain
            # solutes and the list must be left untouched. Replacing the entry
            # unconditionally would enter j twice and drop i, corrupting the
            # proposal distribution over long runs.
            if sj == 0:
                for t in range(n_sol):
                    if solutes[t] == i:
                        solutes[t] = j
                        break
            dE_total += dE
            n_acc += 1

    return dE_total, n_acc / max(n_steps, 1)


@njit(cache=True, fastmath=True)
def _pair_counts(spec, nn1):
    """
    Count first-neighbour bonds by species pair; returns the 4x4 bond matrix N_ab.

    Double-counting convention (a frequent source of error, so stated explicitly):
      raw accumulation  Raw[a,b] = sum_i delta(s_i, a) * sum_k delta(s_nn, b)
      - heterovalent bonds a != b : each bond is counted once, from the a end
                                    -> N_ab = Raw[a,b]
      - homovalent bonds  a == b  : each bond is counted from both ends
                                    -> N_aa = Raw[a,a] / 2
    Dividing the off-diagonal entries by two as well would understate the Cu-In
    bond count, and hence alpha, by a factor of two.
    """
    C = np.zeros((4, 4), dtype=np.float64)
    for i in range(spec.shape[0]):
        si = spec[i]
        for k in range(nn1.shape[1]):
            C[si, spec[nn1[i, k]]] += 1.0
    for a in range(4):
        C[a, a] *= 0.5
    return C


# ==============================================================================
# 5. Simulator
# ==============================================================================
class SphaleriteMC:
    """Three-dimensional lattice Monte Carlo simulator for Cu-In coupled
    substitution in sphalerite."""

    def __init__(self, lattice: SphaleriteLattice, params: HamiltonianParams,
                 x_cu: float = 0.02, x_in: float = 0.02, x_vac: float = 0.0,
                 seed: int = 20260803):
        self.lat = lattice
        self.par = params
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.P1, self.P2, self.SELF = params.build_tables(lattice.a0)
        self.dq = np.ascontiguousarray(DELTA_Q)
        self._init_config(x_cu, x_in, x_vac)

    # ------------------------------------------------------------------
    def _init_config(self, x_cu, x_in, x_vac):
        """Random initial configuration (infinite-temperature limit) at fixed
        composition."""
        N = self.lat.N
        n_cu, n_in, n_vac = int(round(x_cu * N)), int(round(x_in * N)), int(round(x_vac * N))
        spec = np.zeros(N, dtype=np.int8)
        idx = self.rng.permutation(N)
        spec[idx[:n_cu]] = 1
        spec[idx[n_cu:n_cu + n_in]] = 2
        spec[idx[n_cu + n_in:n_cu + n_in + n_vac]] = 3
        self.spec = np.ascontiguousarray(spec)
        self.counts0 = (n_cu, n_in, n_vac)
        self.x = np.array([1 - (n_cu + n_in + n_vac) / N, n_cu / N, n_in / N, n_vac / N])
        self._refresh()

    def _refresh(self):
        self.Q = _anion_charges(self.spec, self.lat.cat_of_an, self.dq)
        self.solutes = np.ascontiguousarray(np.where(self.spec != 0)[0].astype(np.int64))
        self.E = _total_energy(self.spec, self.lat.nn1, self.lat.nn2,
                               self.Q, self.P1, self.P2, self.SELF, self.par.lam)

    # ------------------------------------------------------------------
    def run(self, T_kelvin: float, n_sweeps: int, seed_offset: int = 0,
            validate: bool = True):
        """Run n_sweeps sweeps at temperature T (one sweep = N attempted exchanges)."""
        beta = 1.0 / (KB_EV * T_kelvin)
        n_steps = int(n_sweeps) * self.lat.N
        dE, acc = _mc_run(self.spec, self.Q, self.lat.nn1, self.lat.nn2,
                          self.lat.an_of_cat, self.dq, self.P1, self.P2,
                          self.par.lam, beta, n_steps, self.solutes,
                          self.seed + seed_offset)
        self.E += dE
        if validate:
            self.validate_state()
        return acc

    def validate_state(self, tol: float = 1e-6):
        """
        Run-time integrity checks:
          (1) the solute index list matches the actual set of non-Zn sites,
              with no duplicates and no omissions
          (2) the incrementally maintained Q agrees with a full recomputation
          (3) the incrementally accumulated E agrees with a full recomputation
        """
        true_sol = np.where(self.spec != 0)[0]
        assert np.array_equal(np.sort(self.solutes), true_sol), "solute index list is corrupted"
        Q_ref = _anion_charges(self.spec, self.lat.cat_of_an, self.dq)
        assert np.allclose(self.Q, Q_ref, atol=1e-9), "anion charge table has drifted"
        E_ref = _total_energy(self.spec, self.lat.nn1, self.lat.nn2,
                              Q_ref, self.P1, self.P2, self.SELF, self.par.lam)
        assert abs(self.E - E_ref) < tol * max(1.0, abs(E_ref)), \
            f"energy drift: incremental = {self.E:.9f}, recomputed = {E_ref:.9f}"

    def equilibration_trace(self, T_kelvin: float, n_blocks: int, sweeps_per_block: int,
                            seed_offset: int = 0):
        """Run in blocks, recording energy and order parameter, to diagnose
        convergence."""
        tr = {"sweep": [], "E_per_site": [], "alpha": [], "acc": []}
        for b in range(n_blocks):
            acc = self.run(T_kelvin, sweeps_per_block, seed_offset=seed_offset + b)
            tr["sweep"].append((b + 1) * sweeps_per_block)
            tr["E_per_site"].append(self.E / self.lat.N)
            tr["alpha"].append(self.warren_cowley(2, 1))
            tr["acc"].append(acc)
        return {k: np.array(v) for k, v in tr.items()}

    # ------------------------------------------------------------------
    # Order parameters and structural analysis
    # ------------------------------------------------------------------
    def warren_cowley(self, a: int = 2, b: int = 1) -> float:
        """
        Warren-Cowley short-range order parameter

            alpha_ab = 1 - P(b | first shell of a) / x_b

        alpha < 0  b is enriched around a (association / ordering)
        alpha = 0  random solid solution
        alpha > 0  mutual avoidance
        Defaults are a = In (2), b = Cu (1).
        """
        C = _pair_counts(self.spec, self.lat.nn1)
        n_a = np.count_nonzero(self.spec == a)
        if n_a == 0:
            return np.nan
        z = self.lat.nn1.shape[1]                       # coordination number, 12
        raw_ab = C[a, b] if a != b else 2.0 * C[a, b]   # back to "b neighbours seen from the a end"
        p = raw_ab / (n_a * z)                          # P(b | first shell of a)
        return 1.0 - p / self.x[b]

    def pair_enrichment(self, a: int = 1, b: int = 2) -> float:
        """
        Number of first-neighbour a-b bonds relative to the random-solution
        expectation (enrichment factor R).

            random limit:  N_ab^rand = (N z / 2) * 2 x_a x_b   for a != b
            identity:      R_ab = 1 - alpha_ba, used as a cross-check

        """
        C = _pair_counts(self.spec, self.lat.nn1)
        n_bonds = self.lat.N * self.lat.nn1.shape[1] / 2.0
        expected = n_bonds * (2.0 * self.x[a] * self.x[b] if a != b else self.x[a] ** 2)
        return C[a, b] / expected if expected > 0 else np.nan

    def mean_dq2(self) -> float:
        """Mean squared charge deviation per S tetrahedron, <(Delta Q)^2>: a direct
        measure of how completely local electroneutrality has been achieved."""
        return float(np.mean(self.Q ** 2))

    def cluster_sizes(self) -> np.ndarray:
        """Union-find extraction of solute (Cu/In) clusters connected through the
        first neighbour shell; returns cluster sizes in descending order."""
        mask = (self.spec == 1) | (self.spec == 2)
        sites = np.where(mask)[0]
        pos_of = -np.ones(self.lat.N, dtype=np.int64)
        pos_of[sites] = np.arange(sites.size)
        parent = np.arange(sites.size)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for t, i in enumerate(sites):
            for nb in self.lat.nn1[i]:
                if mask[nb]:
                    ra, rb = find(t), find(pos_of[nb])
                    if ra != rb:
                        parent[ra] = rb
        roots = np.array([find(t) for t in range(sites.size)])
        _, sizes = np.unique(roots, return_counts=True)
        return np.sort(sizes)[::-1]

    def stoichiometry_check(self) -> tuple:
        """Return (n_Cu, n_In, n_vacancy) to verify canonical-ensemble conservation."""
        return tuple(int(np.count_nonzero(self.spec == s)) for s in (1, 2, 3))


# ==============================================================================
# 6. Annealing protocol
# ==============================================================================
def simulated_annealing_scan(lat, par, x_cu, x_in, temps_K,
                             n_equil=1200, n_prod=1200, seed=20260803,
                             verbose=True):
    """
    Sequential cooling protocol. The system is cooled stage by stage from a
    disordered high-temperature state, carrying the configuration over between
    stages, which avoids the configurational freezing and loss of ergodicity that
    a direct quench would produce.

    Returns a dictionary of observables versus temperature.
    """
    mc = SphaleriteMC(lat, par, x_cu=x_cu, x_in=x_in, seed=seed)
    rec = {k: [] for k in ("T", "kT_over_lam", "E_per_site", "alpha_InCu",
                           "R_CuIn", "mean_dq2", "acc", "max_cluster", "mean_cluster")}
    for n, T in enumerate(temps_K):
        mc.run(T, n_equil, seed_offset=1000 * n + 1)          # equilibration
        acc = mc.run(T, n_prod, seed_offset=1000 * n + 2)     # production
        cs = mc.cluster_sizes()
        rec["T"].append(T)
        rec["kT_over_lam"].append(KB_EV * T / par.lam if par.lam > 0 else np.inf)
        rec["E_per_site"].append(mc.E / lat.N)
        rec["alpha_InCu"].append(mc.warren_cowley(2, 1))
        rec["R_CuIn"].append(mc.pair_enrichment(1, 2))
        rec["mean_dq2"].append(mc.mean_dq2())
        rec["acc"].append(acc)
        rec["max_cluster"].append(int(cs[0]) if cs.size else 0)
        rec["mean_cluster"].append(float(cs.mean()) if cs.size else 0.0)
        if verbose:
            print(f"  T={T:7.1f} K | kT/lam={rec['kT_over_lam'][-1]:5.2f} | "
                  f"E/site={rec['E_per_site'][-1]:+8.4f} eV | "
                  f"alpha(In-Cu)={rec['alpha_InCu'][-1]:+8.3f} | "
                  f"R_CuIn={rec['R_CuIn'][-1]:6.2f} | "
                  f"<dQ2>={rec['mean_dq2'][-1]:5.3f} | acc={acc:8.5f}")
    return {k: np.array(v) for k, v in rec.items()}, mc


# ==============================================================================
# 7. Verification and validation suite
# ==============================================================================
def self_test():
    """Unit tests covering geometry, energy bookkeeping, analytic limits and the
    random-solution limit of the order parameters."""
    print("[test] Building an L = 4 lattice ...")
    lat = SphaleriteLattice(4)
    assert lat.N == 256
    print(f"        N = {lat.N}, Z1 = {lat.nn1.shape[1]}, Z2 = {lat.nn2.shape[1]}, "
          f"CN(S) = {lat.cat_of_an.shape[1]}  -> OK")

    par = HamiltonianParams()
    d_nn = lat.a0 / np.sqrt(2)
    print(f"[test] d_NN (cation-cation) = {d_nn:.4f} A")
    print(f"        lambda (theory, eps_r = {par.eps_r}) = {par.lambda_from_dielectric(d_nn):.4f} eV")
    print(f"        Eshelby self-energies = {np.round(par.eshelby_self_energy(lat.a0), 4)} eV")

    mc = SphaleriteMC(lat, par, x_cu=0.05, x_in=0.05, seed=1)
    mc.run(600.0, 200)
    E_ref = _total_energy(mc.spec, lat.nn1, lat.nn2, mc.Q, mc.P1, mc.P2, mc.SELF, par.lam)
    print(f"[test] incremental E = {mc.E:.6f} eV ; recomputed = {E_ref:.6f} eV ; "
          f"deviation = {abs(mc.E - E_ref):.2e} eV")
    assert abs(mc.E - E_ref) < 1e-6 * max(1.0, abs(E_ref)), \
        "incremental dE disagrees with the total energy"
    n_cu, n_in, n_vac = mc.stoichiometry_check()
    assert (n_cu, n_in) == mc.counts0[:2], "composition not conserved in the canonical ensemble"
    print(f"[test] composition conserved: Cu = {n_cu}, In = {n_in}, Vac = {n_vac}  -> OK")

    # Analytic limit: the electrostatic binding energy of a nearest-neighbour
    # Cu-In pair must be exactly -2 lambda.
    spec = np.zeros(lat.N, dtype=np.int8)
    Q = _anion_charges(spec, lat.cat_of_an, DELTA_Q)
    E0 = par.lam * np.sum(Q ** 2)
    i = 0
    # Reference site that is neither a first neighbour of i nor shares a tetrahedron with it.
    far = -1
    for cand in range(1, lat.N):
        if cand in lat.nn1[i]:
            continue
        if np.intersect1d(lat.an_of_cat[i], lat.an_of_cat[cand]).size == 0:
            far = cand
            break
    assert far > 0
    spec[i], spec[far] = 1, 2              # separated Cu and In
    Q = _anion_charges(spec, lat.cat_of_an, DELTA_Q)
    E_sep = par.lam * np.sum(Q ** 2) - E0
    spec[:] = 0
    spec[i], spec[lat.nn1[i, 0]] = 1, 2    # nearest-neighbour Cu-In pair
    Q = _anion_charges(spec, lat.cat_of_an, DELTA_Q)
    E_pair = par.lam * np.sum(Q ** 2) - E0
    print(f"[test] E_c(separated Cu + In) = {E_sep:.4f} eV ; E_c(NN Cu-In pair) = {E_pair:.4f} eV")
    print(f"        binding energy = {E_pair - E_sep:+.4f} eV (analytic {-2*par.lam:+.4f} eV)")
    assert abs((E_pair - E_sep) - (-2 * par.lam)) < 1e-9, "the -2 lambda result is not reproduced"

    # Random limit of the order parameters: a non-interacting system must give
    # R -> 1 and alpha -> 0.
    lat2 = SphaleriteLattice(12)
    par0 = HamiltonianParams(lam=0.0, gamma1=0.0)
    par0.J = np.zeros((4, 4))
    Rs, als = [], []
    for s in range(24):
        m0 = SphaleriteMC(lat2, par0, x_cu=0.02, x_in=0.02, seed=1000 + s)
        Rs.append(m0.pair_enrichment(1, 2))
        als.append(m0.warren_cowley(2, 1))
    Rs, als = np.array(Rs), np.array(als)
    print(f"[test] random limit (24 independent configurations, N = {lat2.N}):")
    print(f"        R_CuIn   = {Rs.mean():.3f} +/- {Rs.std():.3f}   (expected 1)")
    print(f"        alpha    = {als.mean():+.3f} +/- {als.std():.3f}   (expected 0)")
    print(f"        identity R = 1 - alpha, max deviation = {np.max(np.abs(Rs - (1 - als))):.2e}")
    assert abs(Rs.mean() - 1.0) < 0.15, "R departs significantly from 1 in the random limit"
    assert np.max(np.abs(Rs - (1 - als))) < 1e-9, "the identity R = 1 - alpha is violated"

    print("[test] all checks passed")


if __name__ == "__main__":
    t0 = time.time()
    self_test()
    print(f"\nElapsed {time.time() - t0:.1f} s ; Numba = {HAS_NUMBA}")
