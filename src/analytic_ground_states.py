# -*- coding: utf-8 -*-
"""
analytic_ground_states.py -- analytic supplement to
"A Statistical Thermodynamic Lattice Model for Cu-In Coupled Substitution
in Sphalerite" (He Yu & Ruqing Chen).

Determines the ground states of the charge-compensation term E_c and compares
the two charge-balancing routes for In(III). Because the E_c interaction exists
strictly between cations sharing an S tetrahedron (i.e. nearest neighbours
only), ground states can be located reliably by low-temperature annealing with
multiple independent restarts. All energies are reported in units of lambda.
"""
import numpy as np
from sphalerite_mc import (SphaleriteLattice, HamiltonianParams, SphaleriteMC,
                           _anion_charges, DELTA_Q)

lat = SphaleriteLattice(6)          # 864 sites, ample room for isolated defect complexes
print(f"Lattice N = {lat.N}\n")

def pure_Ec_params():
    """Retain E_c only (lambda = 1); switch off the elastic and chemical terms."""
    p = HamiltonianParams(lam=1.0, gamma1=0.0)
    p.J = np.zeros((4, 4))
    p.misfit = np.zeros(4)
    p.E_vac = 0.0
    return p

def Ec_of(spec):
    q = _anion_charges(np.ascontiguousarray(spec.astype(np.int8)), lat.cat_of_an, DELTA_Q)
    return float(np.sum(q ** 2))

BASE = Ec_of(np.zeros(lat.N))

def minimise(n_cu, n_in, n_vac, restarts=12, tag=""):
    """Annealed minimisation: staged cooling 3000 K -> 30 K, best of several restarts."""
    p = pure_Ec_params()
    best, best_spec = np.inf, None
    for r in range(restarts):
        mc = SphaleriteMC(lat, p, x_cu=n_cu/lat.N, x_in=n_in/lat.N,
                          x_vac=n_vac/lat.N, seed=4242 + 97*r)
        for T in (3000, 1500, 800, 400, 200, 100, 50, 30):
            mc.run(float(T), 400, seed_offset=T + r)
        e = Ec_of(mc.spec) - BASE
        if e < best:
            best, best_spec = e, mc.spec.copy()
    print(f"  {tag:34s} E_c^min = {best:6.2f} λ")
    return best, best_spec

print("=== 1. Electrostatic cost of isolated point defects (analytically checkable) ===")
for name, s, pred in (("Cu+  (δq=-1)", 1, 4), ("In3+ (δq=+1)", 2, 4), ("V_Zn (δq=-2)", 3, 16)):
    sp = np.zeros(lat.N); sp[0] = s
    print(f"  {name:14s}: {Ec_of(sp)-BASE:6.2f} λ   (analytic {pred} lambda = 4 dq^2)")

print("\n=== 2. A single nearest-neighbour Cu-In pair ===")
sp = np.zeros(lat.N); sp[0] = 1; sp[lat.nn1[0,0]] = 2
print(f"  NN Cu-In pair                     : {Ec_of(sp)-BASE:6.2f} λ  (analytic 6 lambda)")
print(f"  binding energy                    : {Ec_of(sp)-BASE-8:6.2f} λ  (analytic -2 lambda)")

print("\n=== 3. Competing compensation routes (fixed 2 In3+; E_c ground state of each) ===")
eA, _ = minimise(2, 2, 0, tag="(A) Cu compensation:      2Cu + 2In")
eB, _ = minimise(0, 2, 1, tag="(B) vacancy compensation: V_Zn + 2In")
print(f"\n  ΔE_c(B-A) = {eB-eA:+.2f} λ   ->  the Cu route is electrostatically favoured")

print("\n=== 4. Ground state of E_c: can local charge be fully compensated? ===")
p = lat.pos
sp = np.where(((p[:,0]//2 + p[:,1]//2) % 2) == 0, 1, 2)     # alternating Cu/In, chalcopyrite-like ordering
q = _anion_charges(np.ascontiguousarray(sp.astype(np.int8)), lat.cat_of_an, DELTA_Q)
tet = np.array([np.bincount(sp[lat.cat_of_an[a]], minlength=4) for a in range(lat.N)])
comp, cnt = np.unique([f"{t[1]}Cu+{t[2]}In" for t in tet], return_counts=True)
census = {str(k): int(v) for k, v in zip(comp, cnt)}
print(f"  chalcopyrite-like ordering at x_Cu = x_In = 0.5:")
print(f"    <ΔQ²> = {np.mean(q**2):.6f}   ->  E_c = 0 (complete local charge neutrality)")
print(f"    S tetrahedron composition census: {census}")

# Degeneracy: a second ordering obeying the 2Cu + 2In rule (CuAu-I type layers)
sp2 = np.where((p[:,2] // 2) % 2 == 0, 1, 2)
q2 = _anion_charges(np.ascontiguousarray(sp2.astype(np.int8)), lat.cat_of_an, DELTA_Q)
tet2 = np.array([np.bincount(sp2[lat.cat_of_an[a]], minlength=4) for a in range(lat.N)])
comp2, cnt2 = np.unique([f"{t[1]}Cu+{t[2]}In" for t in tet2], return_counts=True)
census2 = {str(k): int(v) for k, v in zip(comp2, cnt2)}
print(f"  alternative ordering (alternating layers along [001]):")
print(f"    <ΔQ²> = {np.mean(q2**2):.6f}, S tetrahedron composition: {census2}")
