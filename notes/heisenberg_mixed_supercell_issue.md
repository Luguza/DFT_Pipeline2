# `HeisenbergMapper` silently returns wrong exchange parameters when orderings live in different-sized supercells

**Labels:** bug

### Python version

Python 3.12.13

### Pymatgen version

2026.5.4 (reproduced on `master` @ `d6b4ea50a0`, 2026-06-02)

### Operating system version

Rocky Linux 8.10

---

### Current behavior

`HeisenbergMapper` is documented to take a set of low-energy magnetic orderings,
map them onto a classical Heisenberg model, and solve for the exchange
parameters `J_ij`. It implicitly assumes **every ordering lives in the same
supercell**. When the input orderings instead live in *different-sized*
supercells of a common parent cell — which is the normal output of a magnetic
enumeration workflow — `get_exchange()` returns numbers that are silently and
badly wrong. No exception is raised.

This is caused by two independent bugs that both surface only once the supercell
sizes differ:

1. **Site identifiers are built from a single ordering.** In `__init__`:

   ```python
   self.unique_site_ids, self.wyckoff_ids = self._get_unique_sites(ordered_structures[0])
   ```

   `_get_unique_sites(structure)` runs a `SpacegroupAnalyzer` on **only the first
   ordering** and returns one dict mapping that ordering's site indices to
   identifiers. `_get_exchange_df` then reuses that single dict for **every**
   graph:

   ```python
   for k, v in unique_site_ids.items():
       if idx in k:        # idx is a site index in *this* ordering's supercell
           i_index = v
   ```

   For any ordering larger than `ordered_structures[0]`, site indices beyond the
   first cell's size are simply absent from the dict, so `i_index`/`j_index`
   stay `None`, the interaction is mislabeled (`"None-..."`) and dropped. The
   identifiers are also not shared across orderings — the same crystallographic
   orbit can get different ids in different supercells — so the `J_ij` columns
   are not comparable from one row of the fit matrix to the next.

2. **The fit mixes intensive and extensive quantities.** `HeisenbergScreener._do_cleanup`
   normalizes the energies to *per magnetic ion* (intensive):

   ```python
   energies = [e / len(s) for (e, s) in zip(energies, ordered_structures, strict=True)]
   ```

   and `_get_exchange_df` fixes the `E0` (nonmagnetic) column coefficient at `1`.
   But the spin-product interaction columns in the same matrix are accumulated as
   a sum over **the whole supercell** (extensive) — every site, every bond, only
   a global `.div(2)` at the end:

   ```python
   ex_row.loc[sgraph_index, j_ij] -= s_i * s_j   # summed over all sites in the cell
   ...
   ex_mat[j_columns] = ex_mat[j_columns].div(2)
   ```

   With one fixed supercell size this is harmless (the extra factor of `N` is a
   constant). With mixed sizes the left-hand side (per ion) and the columns (per
   supercell) are on different footings, the linear system is inconsistent, and
   the recovered `J`/`E0` are wrong and scale-dependent.

### Expected behavior

The recovered exchange parameters should be independent of the supercell sizes
of the input orderings and should equal the parameters of the underlying
Heisenberg model. Orderings in different supercells of one parent cell should be
a fully supported input (it is the natural output of a magnetic-ordering
enumeration), not an unstated precondition.

### Minimal example

A clean test on a known Hamiltonian: a 2D square lattice of `Fe` with
`E0 = -5.0 eV/ion`, NN coupling `J_nn = +0.012 eV`, NNN coupling
`J_nnn = -0.004 eV`. Energies are assigned analytically from
`E = N * E0 - sum_<ij> J_ab s_i s_j`, so the fit must recover `E0`, `J_nn`,
`J_nnn` exactly, regardless of supercell size. The three orderings deliberately
live in different cells: FM in the 1×1 (1 ion), stripe-AFM in 1×2 (2 ions),
Néel-AFM in 2×2 (4 ions).

```python
import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.analysis.magnetism.heisenberg import HeisenbergMapper

E0, J_NN, J_NNN = -5.0, 0.012, -0.004
NN  = ((1, 0), (-1, 0), (0, 1), (0, -1))
NNN = ((1, 1), (1, -1), (-1, 1), (-1, -1))

def make_structure(spins):
    spins = np.atleast_2d(np.asarray(spins, float))
    n_y, n_x = spins.shape
    lattice = Lattice.from_parameters(n_x, n_y, 10, 90, 90, 90)
    coords = [[x / n_x, y / n_y, 0.5] for y in range(n_y) for x in range(n_x)]
    magmoms = [spins[y, x] for y in range(n_y) for x in range(n_x)]
    return Structure(lattice, ["Fe"] * (n_x * n_y), coords, site_properties={"magmom": magmoms})

def energy(spins):
    spins = np.atleast_2d(np.asarray(spins, float))
    n_y, n_x = spins.shape
    e_ex = 0.0
    for y in range(n_y):
        for x in range(n_x):
            s_i = spins[y, x]
            nn  = sum(s_i * spins[(y + dy) % n_y, (x + dx) % n_x] for dx, dy in NN)
            nnn = sum(s_i * spins[(y + dy) % n_y, (x + dx) % n_x] for dx, dy in NNN)
            e_ex -= 0.5 * (J_NN * nn + J_NNN * nnn)   # 1/2: each bond counted from both ends
    return n_x * n_y * E0 + e_ex

all_spins = [[[1]], [[1], [-1]], [[1, -1], [-1, 1]]]   # FM 1x1, stripe 1x2, Neel 2x2
structures = [make_structure(s) for s in all_spins]
energies   = [energy(s) for s in all_spins]

hm = HeisenbergMapper(structures, energies, cutoff=1.5, tol=0.02)
print("unique_site_ids:", hm.unique_site_ids)
print("exchange:", hm.get_exchange())
```

**Observed (master):**

```
unique_site_ids: {(0,): 0}                 # built from the 1-site FM cell only
exchange: {'E0': -4.968, '0-0-nn': 40.0, '0-0-nnn': -16.0}
```

**Expected:**

```
exchange: {'E0': -5.0, '0-0-nn': 12.0, '0-0-nnn': -4.0}   # J in meV
```

`E0` collapses onto the per-ion energy of one ordering and the `J` values are off
by factors of ~3–4. The result is also unstable under changing which supercells
are supplied.

### Why the current test suite misses this

Every ordering in `tests/files/analysis/magnetic_orderings/Mn3Al.json` already
lives in the same supercell, so both bugs cancel (constant `N`, and every
ordering's site indices are present in the single dict). The failure mode only
appears once the input orderings differ in size, which the existing tests never
exercise.

### Suggested direction for a fix

- Compute one `unique_site_ids` map **per ordering**, anchored to a common
  parent cell, so a given crystallographic orbit receives the same identifier in
  every ordering regardless of supercell size; have `_get_exchange_df` select the
  map for the ordering it is currently iterating. Orderings that are not a
  supercell of the parent should raise rather than be silently mislabeled.
- Put the interaction columns on the **same per-ion footing as the energies**
  (e.g. normalize each row by its site count) so orderings in different-sized
  supercells contribute consistent rows to the linear fit.

### Relevant files

- `src/pymatgen/analysis/magnetism/heisenberg.py`
  - `HeisenbergMapper.__init__` — `_get_unique_sites(ordered_structures[0])`
  - `HeisenbergMapper._get_unique_sites`
  - `HeisenbergMapper._get_exchange_df`
  - `HeisenbergScreener._do_cleanup` — per-ion energy normalization
