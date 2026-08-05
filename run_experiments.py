# -*- coding: utf-8 -*-
"""
run_experiments.py -- numerical experiments for
"A Statistical Thermodynamic Lattice Model for Cu-In Coupled Substitution
in Sphalerite" (He Yu & Ruqing Chen).

  Experiment A : sequential-cooling annealing scan
                 A1 full model | A2 elastic terms only | A3 electrostatics only
  Experiment B : sensitivity to the charge-compensation coefficient lambda,
                 at T = 573 K, three independent seeds per value
  Experiment C : solute cluster statistics and slab projections at 373 K
  Experiment D : equilibration test, quenched versus pre-annealed trajectories

Outputs: results_phase1.csv and Figs. 1-4 as both PDF (vector, for the
manuscript) and PNG (raster, for quick inspection).

Runtime: about 6-7 minutes single-threaded with Numba enabled.
"""
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sphalerite_mc import (SphaleriteLattice, HamiltonianParams, SphaleriteMC,
                           simulated_annealing_scan, KB_EV)

# ---------------- global settings ----------------
L        = 10                 # 10x10x10 cells -> N = 4000 cation sites
X_CU     = 0.02               # 2 at.% Cu
X_IN     = 0.02               # 2 at.% In
N_EQ     = 4000               # equilibration sweeps
N_PR     = 3000               # production sweeps
SEED     = 20260803

TEMPS = np.array([3000, 2400, 2000, 1600, 1300, 1100, 950, 850,
                  773, 700, 640, 573, 500, 450, 400, 373, 320], dtype=float)

t_start = time.time()
lat = SphaleriteLattice(L)
print(f"Lattice: L={L}, N_cation={lat.N}, N_anion={lat.N}, "
      f"Cu={int(X_CU*lat.N)}, In={int(X_IN*lat.N)}\n")

# ==================================================================
# Experiment A: annealing scan
# ==================================================================
print("=" * 78)
print("Experiment A1: full model (lambda = 0.30 eV, J_CuIn = -0.10 eV)")
print("=" * 78)
par_full = HamiltonianParams()
rec_full, mc_full = simulated_annealing_scan(lat, par_full, X_CU, X_IN, TEMPS,
                                             N_EQ, N_PR, seed=SEED)

print("\n" + "=" * 78)
print("Experiment A2: elastic terms only (lambda = 0, J = 0)")
print("=" * 78)
par_ctrl = HamiltonianParams(lam=0.0)
par_ctrl.J = np.zeros((4, 4))
rec_ctrl, mc_ctrl = simulated_annealing_scan(lat, par_ctrl, X_CU, X_IN, TEMPS,
                                             N_EQ, N_PR, seed=SEED)

print("\n" + "=" * 78)
print("Experiment A3: electrostatics only (lambda = 0.30 eV, J = 0)")
print("=" * 78)
par_elec = HamiltonianParams()
par_elec.J = np.zeros((4, 4))
rec_elec, mc_elec = simulated_annealing_scan(lat, par_elec, X_CU, X_IN, TEMPS,
                                             N_EQ, N_PR, seed=SEED)

# ==================================================================
# Experiment B: sensitivity to lambda at T = 573 K (300 C)
# ==================================================================
print("\n" + "=" * 78)
print("Experiment B: lambda sensitivity scan at T = 573 K (300 C)")
print("=" * 78)
lams = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])
N_SEED = 3                                       # three independent seeds per lambda, to average over nucleation stochasticity
ANNEAL = [(2500.0, 800), (1800.0, 800), (1200.0, 800), (900.0, 800), (700.0, 800)]
lam_res = {k: [] for k in ("lam", "alpha", "alpha_sd", "R", "R_sd", "dq2", "dq2_sd")}
for k, lm in enumerate(lams):
    p = HamiltonianParams(lam=float(lm))
    p.J = np.zeros((4, 4))                       # electrostatics only, to isolate a single variable
    a_s, r_s, q_s = [], [], []
    for s in range(N_SEED):
        m = SphaleriteMC(lat, p, x_cu=X_CU, x_in=X_IN, seed=SEED + 100 * k + s)
        for T_a, n_a in ANNEAL:                  # stage-wise annealing, to avoid a quench artefact
            m.run(T_a, n_a, seed_offset=int(T_a) + s)
        m.run(573.0, 2500, seed_offset=1 + s)    # equilibration at the target temperature
        m.run(573.0, 1500, seed_offset=2 + s)    # production
        a_s.append(m.warren_cowley(2, 1))
        r_s.append(m.pair_enrichment(1, 2))
        q_s.append(m.mean_dq2())
    lam_res["lam"].append(lm)
    for nm, v in (("alpha", a_s), ("R", r_s), ("dq2", q_s)):
        lam_res[nm].append(np.mean(v)); lam_res[nm + "_sd"].append(np.std(v))
    print(f"  lambda={lm:4.2f} eV | alpha(In-Cu)={lam_res['alpha'][-1]:+7.2f}+/-{lam_res['alpha_sd'][-1]:5.2f} | "
          f"R_CuIn={lam_res['R'][-1]:6.2f}+/-{lam_res['R_sd'][-1]:5.2f} | "
          f"<dQ2>={lam_res['dq2'][-1]:5.3f}")
lam_res = {k: np.array(v) for k, v in lam_res.items()}

# ==================================================================
# Experiment C: low-temperature configurations
# ==================================================================
print("\n" + "=" * 78)
print("Experiment C: configurations at T = 373 K (100 C)")
print("=" * 78)
cs_full = mc_full.cluster_sizes()
cs_elec = mc_elec.cluster_sizes()
cs_ctrl = mc_ctrl.cluster_sizes()
for nm, cs in (("full       ", cs_full), ("electrostatic", cs_elec), ("elastic only ", cs_ctrl)):
    print(f"  {nm} : clusters={cs.size:4d}, largest={cs[0]:4d}, mean={cs.mean():6.2f}, "
          f"monomer fraction={np.count_nonzero(cs==1)/cs.size:5.2f}")

from sphalerite_mc import _pair_counts
C_full = _pair_counts(mc_full.spec, lat.nn1)
print("\n  Full model first-neighbour bond matrix N_ab (rows/cols = Zn,Cu,In,Vac):")
print(np.array2string(C_full.astype(int)))
z = 12; n_in = int(np.count_nonzero(mc_full.spec == 2))
print(f"  In first-shell composition: Cu={C_full[2,1]/(n_in*z):.3f}, "
      f"In={2*C_full[2,2]/(n_in*z):.3f}, Zn={C_full[2,0]/(n_in*z):.3f} "
      f"(sum={{:.3f}}, expected 1.000)".format((C_full[2,1]+2*C_full[2,2]+C_full[2,0])/(n_in*z)))
n_bonds = lat.N * 12 / 2
print(f"  Random expectation for Cu-In bonds = {n_bonds*2*X_CU*X_IN:.1f}; observed = {C_full[1,2]:.0f}")

# ==================================================================
# Experiment D: ergodicity / hysteresis test at 573 K.
#   Only if two independent trajectories converge to the same alpha can the
#   low-temperature results be read as equilibrium rather than kinetic arrest.
# ==================================================================
print("\n" + "=" * 78)
print("Experiment D: ergodicity test at 573 K, quenched versus pre-annealed start")
print("=" * 78)
p = HamiltonianParams()
mc_quench = SphaleriteMC(lat, p, x_cu=X_CU, x_in=X_IN, seed=777)   # direct quench from a random configuration
tr_q = mc_quench.equilibration_trace(573.0, n_blocks=12, sweeps_per_block=1000, seed_offset=10)

mc_slow = SphaleriteMC(lat, p, x_cu=X_CU, x_in=X_IN, seed=778)
for T in (2000.0, 1400.0, 1000.0, 800.0):                          # pre-annealing
    mc_slow.run(T, 1500, seed_offset=int(T))
tr_s = mc_slow.equilibration_trace(573.0, n_blocks=12, sweeps_per_block=1000, seed_offset=20)

print(f"  quenched     alpha: start {tr_q['alpha'][0]:+.3f} -> final {tr_q['alpha'][-1]:+.3f} "
      f"(E/site {tr_q['E_per_site'][-1]:+.5f} eV, acc={tr_q['acc'][-1]:.5f})")
print(f"  pre-annealed alpha: start {tr_s['alpha'][0]:+.3f} -> final {tr_s['alpha'][-1]:+.3f} "
      f"(E/site {tr_s['E_per_site'][-1]:+.5f} eV, acc={tr_s['acc'][-1]:.5f})")
gap = abs(tr_q['alpha'][-1] - tr_s['alpha'][-1])
print(f"  difference between final alpha values = {gap:.3f}  ->  "
      f"{'converged; treat as equilibrium' if gap < 0.6 else 'hysteresis; kinetically arrested'}")

# ==================================================================
# Figures
# ==================================================================
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8,
                     "figure.dpi": 150, "savefig.dpi": 200})
CB = {"full": "#c0392b", "elec": "#2980b9", "ctrl": "#7f8c8d"}
plt.rcParams["pdf.fonttype"] = 42          # embed TrueType, keeps text editable
plt.rcParams["ps.fonttype"] = 42

def save_fig(fig, stem):
    """Write both a vector PDF (for the manuscript) and a raster PNG."""
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight")

# ---- Fig. 1: annealing scan ----
fig, ax = plt.subplots(2, 2, figsize=(9.0, 6.4))
band = dict(color="#f1c40f", alpha=0.18, zorder=0)

def _plot(a, key, ylabel, logx=False):
    a.axvspan(373, 773, **band)
    for rec, lab, c in ((rec_full, "Full model  (λ=0.30, J≠0)", CB["full"]),
                        (rec_elec, "Electrostatic only (λ=0.30, J=0)", CB["elec"]),
                        (rec_ctrl, "Elastic only (λ=0, J=0)", CB["ctrl"])):
        a.plot(rec["T"], rec[key], "o-", ms=3.5, lw=1.3, color=c, label=lab)
    a.set_xlabel("Temperature (K)")
    a.set_ylabel(ylabel)
    a.grid(alpha=0.25, lw=0.5)

_plot(ax[0, 0], "alpha_InCu", r"Warren–Cowley $\alpha_{\rm In-Cu}$")
ax[0, 0].axhline(0, color="k", lw=0.7, ls="--")
ax[0, 0].legend(fontsize=6.8, loc="lower right")
ax[0, 0].set_title("(a) Cu short-range order around In", fontsize=9, loc="left")

_plot(ax[0, 1], "R_CuIn", r"$R_{\rm Cu-In}$ = $N_{\rm CuIn}$ / random")
ax[0, 1].axhline(1, color="k", lw=0.7, ls="--")
ax[0, 1].set_yscale("log")
ax[0, 1].set_title("(b) Cu–In pair enrichment factor", fontsize=9, loc="left")

_plot(ax[1, 0], "mean_dq2", r"$\langle \Delta Q_a^2 \rangle$ per S tetrahedron")
ax[1, 0].set_title("(c) Local charge imbalance", fontsize=9, loc="left")

_plot(ax[1, 1], "E_per_site", r"$H/N$ (eV per cation site)")
ax[1, 1].set_title("(d) Internal energy", fontsize=9, loc="left")

fig.suptitle("Fig. 1  Simulated annealing of Cu–In coupled substitution in sphalerite "
             f"(L={L}, N={lat.N}, $x_{{Cu}}=x_{{In}}=2$ at.%)\n"
             "yellow band = typical hydrothermal ore-forming window (100–500 °C)",
             fontsize=9.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
save_fig(fig, "fig1_annealing")
plt.close(fig)

# ---- Fig. 2: lambda scan ----
fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.2))
ax[0].errorbar(lam_res["lam"], lam_res["alpha"], yerr=lam_res["alpha_sd"],
               fmt="s-", color=CB["elec"], ms=4, capsize=2, lw=1.2)
ax[0].axhline(0, color="k", lw=0.7, ls="--")
ax[0].axvspan(0.227, 0.368, color="#2ecc71", alpha=0.18)
ax[0].set_xlabel(r"$\lambda$ (eV)"); ax[0].set_ylabel(r"$\alpha_{\rm In-Cu}$")
ax[0].set_title(r"(a) SRO vs charge penalty @573 K", fontsize=9, loc="left")
ax[0].grid(alpha=0.25, lw=0.5)
ax[1].errorbar(lam_res["lam"], lam_res["R"], yerr=lam_res["R_sd"],
               fmt="s-", color=CB["full"], ms=4, capsize=2, lw=1.2)
ax[1].axhline(1, color="k", lw=0.7, ls="--")
ax[1].axvspan(0.227, 0.368, color="#2ecc71", alpha=0.18,
              label=r"$\lambda$ from $\varepsilon_\infty$–$\varepsilon_0$ of ZnS")
ax[1].set_yscale("log")
ax[1].set_xlabel(r"$\lambda$ (eV)"); ax[1].set_ylabel(r"$R_{\rm Cu-In}$")
ax[1].set_title("(b) Pair enrichment vs charge penalty", fontsize=9, loc="left")
ax[1].legend(fontsize=7); ax[1].grid(alpha=0.25, lw=0.5)
fig.suptitle(r"Fig. 2  Sensitivity to the charge-compensation coefficient $\lambda$ "
             "(electrostatic term only, J = 0)", fontsize=9.5)
fig.tight_layout(rect=[0, 0, 1, 0.90])
save_fig(fig, "fig2_lambda_scan")
plt.close(fig)

# ---- Fig. 3: cluster distribution and slab projections ----
fig = plt.figure(figsize=(9.0, 3.3))
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.0])

axc = fig.add_subplot(gs[0])
mx = max(cs_full[0], cs_ctrl[0])
bins = np.arange(0.5, mx + 1.5)
axc.hist(cs_ctrl, bins=bins, color=CB["ctrl"], alpha=0.75, label="Elastic only (λ=0)")
axc.hist(cs_full, bins=bins, color=CB["full"], alpha=0.65, label="Full model")
axc.set_yscale("log"); axc.set_xlabel("cluster size (n. of solute atoms)")
axc.set_ylabel("count"); axc.legend(fontsize=7)
axc.set_title("(a) Solute cluster-size distribution @373 K", fontsize=9, loc="left")

# Slab projection along [001] onto the xy plane
def snapshot(ax_, mc, title):
    p, s = mc.lat.pos, mc.spec
    sel = p[:, 2] < 8                      # slab thickness, in units of a0/4
    ax_.scatter(p[sel & (s == 0), 0], p[sel & (s == 0), 1], s=3,
                c="#dfe6e9", marker="o", lw=0)
    ax_.scatter(p[sel & (s == 1), 0], p[sel & (s == 1), 1], s=42,
                c="#e67e22", marker="o", lw=0.4, ec="k", label="Cu$^+$")
    ax_.scatter(p[sel & (s == 2), 0], p[sel & (s == 2), 1], s=42,
                c="#8e44ad", marker="^", lw=0.4, ec="k", label="In$^{3+}$")
    ax_.set_aspect("equal"); ax_.set_xticks([]); ax_.set_yticks([])
    ax_.set_title(title, fontsize=9, loc="left")

ax1 = fig.add_subplot(gs[1]); snapshot(ax1, mc_ctrl, "(b) Elastic only: random solid solution")
ax2 = fig.add_subplot(gs[2]); snapshot(ax2, mc_full, "(c) Full model: Cu–In clusters")
ax2.legend(fontsize=7, loc="upper right", framealpha=0.9)
fig.suptitle("Fig. 3  Emergent Cu–In clustering at 373 K (slab projection along [001])",
             fontsize=9.5)
fig.tight_layout(rect=[0, 0, 1, 0.90])
save_fig(fig, "fig3_cluster")
plt.close(fig)

# ---- Fig. 4: ergodicity test ----
fig, ax = plt.subplots(1, 2, figsize=(7.6, 2.9))
ax[0].plot(tr_q["sweep"], tr_q["alpha"], "o-", ms=3, color="#c0392b", label="quenched start")
ax[0].plot(tr_s["sweep"], tr_s["alpha"], "s-", ms=3, color="#2980b9", label="slow-cooled start")
ax[0].set_xlabel("MC sweeps at 573 K"); ax[0].set_ylabel(r"$\alpha_{\rm In-Cu}$")
ax[0].legend(fontsize=7); ax[0].grid(alpha=0.25, lw=0.5)
ax[0].set_title("(a) Order-parameter convergence", fontsize=9, loc="left")
ax[1].plot(tr_q["sweep"], tr_q["E_per_site"], "o-", ms=3, color="#c0392b")
ax[1].plot(tr_s["sweep"], tr_s["E_per_site"], "s-", ms=3, color="#2980b9")
ax[1].set_xlabel("MC sweeps at 573 K"); ax[1].set_ylabel("H/N (eV)")
ax[1].grid(alpha=0.25, lw=0.5)
ax[1].set_title("(b) Energy convergence", fontsize=9, loc="left")
fig.suptitle("Fig. 4  Ergodicity test: two independent starting configurations at 573 K",
             fontsize=9.5)
fig.tight_layout(rect=[0, 0, 1, 0.89])
save_fig(fig, "fig4_ergodicity")
plt.close(fig)

# ---- CSV ----
import csv
with open("results_phase1.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["experiment", "T_K", "kT_over_lambda", "E_per_site_eV",
                "alpha_In_Cu", "R_CuIn", "mean_dQ2", "acc_rate",
                "max_cluster", "mean_cluster"])
    for name, rec in (("full", rec_full), ("electrostatic_only", rec_elec), ("control", rec_ctrl)):
        for k in range(len(rec["T"])):
            w.writerow([name, rec["T"][k], rec["kT_over_lam"][k], rec["E_per_site"][k],
                        rec["alpha_InCu"][k], rec["R_CuIn"][k], rec["mean_dq2"][k],
                        rec["acc"][k], rec["max_cluster"][k], rec["mean_cluster"][k]])
    for k in range(len(lam_res["lam"])):
        w.writerow(["lambda_scan_573K", 573.0, "", "",
                    lam_res["alpha"][k], lam_res["R"][k], lam_res["dq2"][k],
                    "", "", f"lambda={lam_res['lam'][k]}"])

print(f"\nDone. Total elapsed {time.time() - t_start:.1f} s")
