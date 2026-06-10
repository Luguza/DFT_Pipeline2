#!/usr/bin/env python
"""End-to-end round-trip test for pymatgen's HeisenbergMapper on real MnO.

Where test_heisenberg_fit.py uses a hand-built J1-J2 square lattice, this script
exercises the *production* code path on a real material:

1. Build the rocksalt MnO primitive cell (1 Mn + 1 O, conventional a = 4.445 Ang;
   the mp-19006 geometry), then enumerate collinear magnetic orderings with
   pymatgen's MagneticStructureEnumerator -- the same engine that drives
   atomate2's MagneticOrderingsMaker.
2. Assign each ordering a *total* energy from a KNOWN Heisenberg Hamiltonian

       E_tot = N_mag * e0_mag + N_non * e0_non  -  sum_<ij> J_ab s_i s_j

   with the magnetic (Mn) and nonmagnetic (O) ions carrying *different* per-ion
   references, computed with an independent neighbour sum over the Mn sublattice
   (so it does NOT rely on the mapper's own neighbour enumeration). Because the
   mapper normalises per magnetic ion and MnO is 1 O : 1 Mn, it folds both
   references into one fitted E0 = e0_mag + e0_non -- a genuine end-to-end check.
3. Feed (structures, total energies) to HeisenbergMapper and fit.

The Mn sublattice of MnO is fcc: every Mn has 12 nearest neighbours at a/sqrt(2)
(coupling J_nn) and 6 next-nearest at a (coupling J_nnn); these shell distances
and the search cutoff are hard-coded for MnO (a ~ 4.445 Ang).
The enumerator returns orderings with N_mag in {1, 2, 4} -- i.e. mixed supercell
sizes *by construction* (heisenberg_energy_units.tex, Sec. 8). Two fits are run:

  * FULL SET (the raw enumerator output, mixed N_mag): ill-posed, recovers
    neither e0 nor J -- printed as a demonstration, NOT asserted.
  * COMMON-N_mag SUBSET (orderings that share one supercell size): exact, returns
    E0 = e0_mag + e0_non and J_ab = J_ab_phys / N_mag -- this is the asserted test.

Run:    .venv/bin/python test_heisenberg_mno.py
Pytest: .venv/bin/python -m pytest test_heisenberg_mno.py
"""

from __future__ import annotations

import sys
import warnings
from collections import Counter

import numpy as np

warnings.filterwarnings("ignore")

from pymatgen.core import Lattice, Structure  # noqa: E402
from pymatgen.analysis.magnetism.analyzer import (  # noqa: E402
    CollinearMagneticStructureAnalyzer,
    MagneticStructureEnumerator,
)
from pymatgen.analysis.magnetism.heisenberg import HeisenbergMapper  # noqa: E402

# --- physical "truth" -------------------------------------------------------
# e0_* are arbitrary per-ion references (not real MnO DFT energies); only the
# round trip matters. Mn and O get *different* per-ion energies to show that the
# mapper folds all nonmagnetic energy into a single E0. Signs follow Eq. 1
# (E = e0 - sum J s_i s_j), so J > 0 is ferromagnetic; MnO has weak J_nn and a
# strong antiferromagnetic J_nnn (the 180 deg Mn-O-Mn superexchange).
E0_MAG = -5.0           # eV per magnetic ion (Mn)
E0_NONMAG = -3.0        # eV per nonmagnetic ion (O) -- a *different* reference
JNN_PHYS = 0.0015       # nearest-neighbour exchange      = +1.5 meV
JNNN_PHYS = -0.0080     # next-nearest-neighbour exchange =  -8.0 meV

# --- MnO Mn-sublattice geometry (fcc), hard-coded --------------------------
# a_conv ~ 4.445 Ang: the first two Mn-Mn shells sit at a/sqrt(2) and a; the 3rd
# shell (a*sqrt(3/2) ~ 5.44) falls outside CUTOFF.
D_NN = 3.143            # 1st shell: 12 Mn at a/sqrt(2)  -> J_nn
D_NNN = 4.445           # 2nd shell:  6 Mn at a          -> J_nnn
CUTOFF = 5.0            # NN search cutoff: captures nn + nnn, excludes the 3rd shell
DTOL = 0.1              # Ang tolerance for classifying a bond as nn / nnn


def mno_primitive() -> Structure:
    """Rocksalt MnO primitive cell: 1 Mn + 1 O, conventional a = 4.445 Ang.

    fcc primitive lattice vectors (so the Mn sublattice is fcc); O sits in the
    octahedral hole at the cell-diagonal midpoint. This is the mp-19006 geometry,
    hard-coded so the test needs no network access.
    """
    a = 4.445
    lattice = Lattice([[0.0, a / 2, a / 2],
                       [a / 2, 0.0, a / 2],
                       [a / 2, a / 2, 0.0]])
    return Structure(lattice, ["Mn", "O"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])


def magnetic_sublattice(struct: Structure) -> Structure:
    """Mn-only sublattice carrying a 'magmom' site property (the +-5 pattern).

    This is the same reduction HeisenbergScreener performs internally, so the
    N_mag used here matches the mapper's normalisation.
    """
    cmsa = CollinearMagneticStructureAnalyzer(struct, overwrite_magmom_mode="none")
    return cmsa.get_structure_with_only_magnetic_atoms()


def total_energy(struct: Structure) -> tuple[float, int, int, int, int]:
    """Total energy (eV) of one ordering, by independent site + bond sums.

        E = N_mag * e0_mag + N_non * e0_non - 1/2 sum_i sum_{j in nn/nnn} J s_i s_j

    Magnetic (Mn) and nonmagnetic (O) ions carry different per-ion references.
    get_all_neighbors lists every bond from both endpoints, so the 1/2 removes
    the double count. Returns (E_tot, N_mag, N_non, z_nn, z_nnn) where z_* are the
    per-ion coordination numbers (12 and 6 for fcc -- a built-in sanity check).
    """
    msub = magnetic_sublattice(struct)
    mag = np.asarray(msub.site_properties["magmom"], dtype=float)
    n_mag = len(msub)
    n_non = len(struct) - n_mag          # nonmagnetic (O) ions in this supercell
    e_nn = e_nnn = 0.0
    n_nn = n_nnn = 0
    for i, neighbours in enumerate(msub.get_all_neighbors(CUTOFF)):
        for nb in neighbours:
            sj = mag[nb.index]
            if abs(nb.nn_distance - D_NN) < DTOL:
                e_nn += -0.5 * JNN_PHYS * mag[i] * sj
                n_nn += 1
            elif abs(nb.nn_distance - D_NNN) < DTOL:
                e_nnn += -0.5 * JNNN_PHYS * mag[i] * sj
                n_nnn += 1
    e_tot = n_mag * E0_MAG + n_non * E0_NONMAG + e_nn + e_nnn
    return e_tot, n_mag, n_non, n_nn // n_mag, n_nnn // n_mag


def fit(structures: list[Structure], energies: list[float]) -> dict:
    """Run the mapper and print its design matrix; return get_exchange() dict."""
    hm = HeisenbergMapper(structures, energies, cutoff=CUTOFF, tol=0.05)
    ex = hm.get_exchange()
    print("  unique_site_ids:", hm.unique_site_ids)
    print("  dists:", {k: round(float(v), 3) for k, v in hm.dists.items()})
    print("  ex_mat:\n    " + hm.ex_mat.to_string().replace("\n", "\n    "))
    return ex


def enumerate_mno(primitive: Structure):
    """Enumerate orderings and tag each with its independently computed energy.

    No magmom hint is given: pymatgen recognises Mn as magnetic and assigns its
    built-in high-spin default (5 mu_B). This matches the real pipeline
    (default_magmoms=None) and yields the same orderings as an explicit hint.
    """
    enum = MagneticStructureEnumerator(primitive, default_magmoms=None)

    structures, energies, n_mags = [], [], []
    print(f"  {'idx':>3} {'origin':<10} {'N_mag':>5} {'N_O':>4} {'z_nn':>4} {'z_nnn':>5} {'E_tot [eV]':>13}")
    for idx, (s, origin) in enumerate(zip(enum.ordered_structures, enum.ordered_structure_origins)):
        e_tot, n_mag, n_non, z_nn, z_nnn = total_energy(s)
        structures.append(s)
        energies.append(e_tot)
        n_mags.append(n_mag)
        print(f"  {idx:>3} {origin:<10} {n_mag:>5} {n_non:>4} {z_nn:>4} {z_nnn:>5} {e_tot:>13.6f}")
    return structures, energies, n_mags


def run(primitive: Structure) -> bool:
    print(
        f"physical truth: e0_mag(Mn) = {E0_MAG:+.3f} eV, e0_non(O) = {E0_NONMAG:+.3f} eV, "
        f"Jnn = {JNN_PHYS * 1e3:+.2f} meV, Jnnn = {JNNN_PHYS * 1e3:+.2f} meV"
    )
    print(
        f"  MnO is 1 O : 1 Mn, so the per-magnetic-ion fit reports one "
        f"E0 = e0_mag + e0_non = {E0_MAG + E0_NONMAG:+.3f} eV\n"
    )

    print(f"=== hard-coded {primitive.composition.reduced_formula} primitive cell (cf. mp-19006) ===")
    print(f"  lattice abc={[round(x, 4) for x in primitive.lattice.abc]} "
          f"angles={[round(x, 1) for x in primitive.lattice.angles]}  nsites={len(primitive)}")
    print(f"  Mn fcc shells (hard-coded): d_nn={D_NN:.3f}, d_nnn={D_NNN:.3f} Ang; "
          f"search cutoff={CUTOFF:.3f} Ang\n")

    print("=== enumerate MnO orderings + assign known-Hamiltonian energies ===")
    structures, energies, n_mags = enumerate_mno(primitive)

    # --- demonstration only (NOT asserted): the raw, mixed-N_mag enumerator set
    print("\n=== fit on the FULL enumerator output (mixed N_mag -> ill-posed) ===")
    ex_full = fit(structures, energies)
    print("  fitted :", {k: round(v, 5) for k, v in ex_full.items()})
    print("  NOTE: not expected to match -- MagneticStructureEnumerator mixes")
    print("        supercell sizes by construction (heisenberg_energy_units.tex,")
    print("        Sec. 8), which makes the linear fit ill-posed.")

    # --- asserted test: a subset of orderings that share one supercell size ---
    # Need >= 3 orderings to solve for (E0, Jnn, Jnnn); prefer the larger cell.
    sizes = Counter(n_mags)
    candidates = [n for n, c in sizes.items() if c >= 3]
    if not candidates:
        print("\n  (no N_mag group with >=3 orderings; skipping asserted fit)")
        return False
    common_n = max(candidates)
    idxs = [i for i, n in enumerate(n_mags) if n == common_n]

    print(f"\n=== fit on the COMMON N_mag={common_n} subset (exact recovery) ===")
    sub_structures = [structures[i] for i in idxs]
    sub_energies = [energies[i] for i in idxs]
    ex = fit(sub_structures, sub_energies)

    expect = {
        # MnO is 1 O : 1 Mn, so per magnetic ion the O reference adds in full:
        "E0": E0_MAG + E0_NONMAG,               # eV (left unscaled by get_exchange)
        "0-0-nn": JNN_PHYS / common_n * 1e3,    # meV / mag ion
        "0-0-nnn": JNNN_PHYS / common_n * 1e3,  # meV / mag ion
    }
    got = {"E0": ex["E0"], "0-0-nn": ex["0-0-nn"], "0-0-nnn": ex["0-0-nnn"] }
    print("  fitted :", {k: round(v, 6) for k, v in got.items()})
    print("  expect :", {k: round(v, 6) for k, v in expect.items()},
          "  (E0 = e0_mag+e0_non; Jnn,Jnnn = J_phys / N_mag)")
    ok = all(np.isclose(got[k], expect[k], atol=1e-4) for k in expect)
    print("  ->", "PASS" if ok else "FAIL")
    return ok


def test_heisenberg_mno_roundtrip():
    """Pytest entry point: the common-N_mag fit recovers e0 and J_phys/N_mag."""
    assert run(mno_primitive())


def main() -> int:
    return 0 if run(mno_primitive()) else 1


if __name__ == "__main__":
    sys.exit(main())
