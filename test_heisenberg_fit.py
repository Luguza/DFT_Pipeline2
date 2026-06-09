#!/usr/bin/env python
"""Synthetic round-trip test for pymatgen's HeisenbergMapper.

Strategy
--------
1. Build magnetic orderings on a J1-J2 square lattice (one magnetic ion per
   site, |s| = 1): ferromagnet (FM), row-striped AFM (Stripe), and checkerboard
   Neel AFM.
2. Assign each ordering a *total* energy from a KNOWN Heisenberg Hamiltonian

       E_tot = N_cell * e0  -  sum_<ij> J_ab s_i s_j        (Eq. 1 of the note)

   evaluated analytically on the periodic lattice -- i.e. independent of
   pymatgen's own neighbour enumeration, so this exercises the full mapper
   pipeline end to end.
3. Feed (structures, total energies) to HeisenbergMapper and fit.
4. Assert the fitted couplings equal J_phys / N_mag and E0 == e0 -- the
   per-magnetic-ion quantities the mapper reports (see
   heisenberg_energy_units.tex).

Recovery is exact only when every ordering shares ONE common supercell, so that
N_mag is identical across orderings (the standard enumeration workflow). That is
the asserted test. A second block *demonstrates* the documented failure when
minimal cells of different sizes are mixed (Sec. 9 of the note); it is printed,
not asserted.

Run:  .venv/bin/python test_heisenberg_fit.py
"""

from __future__ import annotations

import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from pymatgen.core import Lattice, Structure  # noqa: E402
from pymatgen.analysis.magnetism.heisenberg import HeisenbergMapper  # noqa: E402

# --- physical "truth" (eV); nu = 1 magnetic ion per primitive cell -----------
E0_PRIM = -5.0       # nonmagnetic reference energy per primitive cell
JNN_PHYS = 0.012     # nearest-neighbour exchange      = +12 meV
JNNN_PHYS = -0.004   # next-nearest-neighbour exchange =  -4 meV

A = 1.0              # lattice constant (Ang): NN = A = 1.0, NNN = A*sqrt2 ~ 1.414
CUTOFF = 1.5 * A     # captures nn and nnn, excludes the next shell at 2.0
CVAC = 10.0          # vacuum along c to isolate the 2D plane

NN = [(1, 0), (-1, 0), (0, 1), (0, -1)]
NNN = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def spin_grid(kind: str, lx: int, ly: int) -> np.ndarray:
    """Periodic +-1 spin pattern on an (ly, lx) grid."""
    x = np.arange(lx)[None, :]
    y = np.arange(ly)[:, None]
    if kind == "FM":
        g = np.ones((ly, lx))
    elif kind == "Stripe":  # rows alternate along y
        g = np.where((y % 2) == 0, 1.0, -1.0) * np.ones((ly, lx))
    elif kind == "Neel":  # checkerboard
        g = np.where(((x + y) % 2) == 0, 1.0, -1.0)
    else:
        raise ValueError(f"unknown ordering {kind!r}")
    return g.astype(float)


def analytic_energy(g: np.ndarray) -> float:
    """Total Heisenberg energy of a periodic spin grid, in eV.

    E = N_cell * e0 - 1/2 sum_i sum_{j in NN/NNN} J s_i s_j   (the 1/2 removes
    the double count from summing every bond from both endpoints).
    """
    ly, lx = g.shape
    e_ex = 0.0
    for y in range(ly):
        for x in range(lx):
            si = g[y, x]
            s_nn = sum(si * g[(y + dy) % ly, (x + dx) % lx] for dx, dy in NN)
            s_nnn = sum(si * g[(y + dy) % ly, (x + dx) % lx] for dx, dy in NNN)
            e_ex += -0.5 * (JNN_PHYS * s_nn + JNNN_PHYS * s_nnn)
    return lx * ly * E0_PRIM + e_ex


def build_structure(g: np.ndarray) -> Structure:
    """Square-lattice supercell carrying the grid's magmoms as 'magmom'."""
    ly, lx = g.shape
    latt = Lattice.from_parameters(lx * A, ly * A, CVAC, 90, 90, 90)
    coords, species, magmoms = [], [], []
    for y in range(ly):
        for x in range(lx):
            coords.append([x / lx, y / ly, 0.5])
            species.append("Fe")
            magmoms.append(float(g[y, x]))
    return Structure(latt, species, coords, site_properties={"magmom": magmoms})


def make_case(cells: dict[str, tuple[int, int]]):
    """cells maps ordering -> (lx, ly). Returns (structures, energies, info)."""
    structures, energies, info = [], [], []
    for kind, (lx, ly) in cells.items():
        g = spin_grid(kind, lx, ly)
        structures.append(build_structure(g))
        energies.append(analytic_energy(g))
        info.append((kind, lx * ly, energies[-1]))
    return structures, energies, info


def coupling(ex: dict, suffix: str) -> float:
    """Pull the single coupling whose label ends with suffix (e.g. '-nn')."""
    hits = [k for k in ex if k.endswith(suffix)]
    if len(hits) != 1:
        raise KeyError(f"expected exactly one {suffix!r} coupling, got {hits}")
    return ex[hits[0]]


def run(cells: dict[str, tuple[int, int]]):
    structures, energies, info = make_case(cells)
    for kind, n, e in info:
        print(f"  {kind:7s} cell={cells[kind]} N={n}  E_tot={e:+.6f} eV")
    hm = HeisenbergMapper(structures, energies, cutoff=CUTOFF, tol=0.02)
    ex = hm.get_exchange()
    print("  unique_site_ids:", hm.unique_site_ids)
    print("  ex_mat:\n    " + hm.ex_mat.to_string().replace("\n", "\n    "))
    return ex


def main() -> int:
    print(
        f"physical truth: e0 = {E0_PRIM:+.3f} eV, "
        f"Jnn = {JNN_PHYS * 1e3:+.1f} meV, Jnnn = {JNNN_PHYS * 1e3:+.1f} meV\n"
    )

    # --- asserted test: one common 2x2 supercell for every ordering (N = 4) --
    print("=== common 2x2 supercell (N = 4 for all orderings) ===")
    n_mag = 4
    ex = run({"FM": (2, 2), "Stripe": (2, 2), "Neel": (2, 2)})
    expect = {
        "nn": JNN_PHYS / n_mag * 1e3,    # meV / mag ion
        "nnn": JNNN_PHYS / n_mag * 1e3,  # meV / mag ion
        "E0": E0_PRIM,                   # eV (left unscaled by get_exchange)
    }
    got = {"nn": coupling(ex, "-nn"), "nnn": coupling(ex, "-nnn"), "E0": ex["E0"]}
    print("  fitted :", {k: round(v, 6) for k, v in got.items()})
    print("  expect :", {k: round(v, 6) for k, v in expect.items()},
          "  (Jnn,Jnnn = J_phys/N)")
    ok = all(np.isclose(got[k], expect[k], atol=1e-6) for k in expect)
    print("  ->", "PASS" if ok else "FAIL")

    # --- demonstration only (NOT asserted): mixed minimal cells --------------
    print("\n=== mixed minimal cells: FM 1x1, Stripe 1x2, Neel 2x2 ===")
    ex_mixed = run({"FM": (1, 1), "Stripe": (1, 2), "Neel": (2, 2)})
    print("  fitted :", {k: round(v, 6) for k, v in ex_mixed.items()})
    print("  NOTE: not expected to match J_phys/N -- mixing supercell sizes")
    print("        breaks the fit (heisenberg_energy_units.tex, Sec. 9):")
    print("        unique_site_ids is built only from the first structure, so")
    print("        bonds touching higher site indices are silently dropped.")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
