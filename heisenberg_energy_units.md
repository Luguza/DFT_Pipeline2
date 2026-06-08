# Energy units through pymatgen's `HeisenbergMapper`

**Scope.** This note traces what happens to the energy — and its units — from the
moment we hand structures and **total energies** to pymatgen's `HeisenbergMapper`,
through the construction of the Heisenberg design matrix, the linear fit itself,
and finally into the `HeisenbergModel` object that gets written out. The upstream
glue (pulling structures/energies out of the magnetic-orderings document and
reconstructing total energies) is deliberately omitted: we start where the class's
public contract starts, i.e. assuming we already have

* a list of magnetic structures (each with a `magmom` site property), and
* a list of **total energies** `E_k^tot` (eV) — the energy of the entire
  supercell of ordering `k`, summed over all of its atoms.

All file/line references are to
`.venv/lib/python3.12/site-packages/pymatgen/analysis/magnetism/heisenberg.py`.

---

## Notation (fixed throughout)

| Symbol | Meaning | Unit |
|---|---|---|
| `E_k^tot` | total DFT energy of ordering `k` (whole supercell) | eV |
| `N_k^mag` | number of **magnetic ions** in ordering `k` | — (count) |
| `ε_k` | energy of ordering `k` normalised per magnetic ion, `ε_k = E_k^tot / N_k^mag` | eV / mag ion |
| `s_i` | local magnetic moment (`magmom`) on magnetic ion `i` | μ_B |
| `J_ab` | exchange constant for neighbour class `ab` (nn/nnn/nnnn between unique sublattices `a`,`b`) | see §4 |
| `E_0` | spin-independent (nonmagnetic) reference energy | eV / mag ion |

The Heisenberg energy model the code assumes (collinear; moments enter as scalars):

```
E^tot = E_0 − Σ_⟨i,j⟩  J_ij · s_i · s_j
```

where the sum runs over unique neighbour bonds `⟨i,j⟩`.

> **Important convention.** Nowhere in this pipeline is energy normalised by the
> total atom count. The only normalisation that ever happens is division by the
> number of **magnetic ions**, `N_k^mag`. So the natural energy unit inside the
> mapper is **eV / mag ion**, and the fitted couplings come out **per mag ion**,
> *not* per (total) atom — despite what the source docstrings call "meV/atom".

---

## Step 1 — Ingestion and normalisation: total energy → eV / mag ion

`HeisenbergMapper.__init__` immediately routes the inputs through
`HeisenbergScreener`, whose `_do_cleanup` does the unit-relevant work
(`heisenberg.py:696`).

1. **Strip to the magnetic sublattice** (`heisenberg.py:717-722`). Each structure
   is reduced to magnetic ions only via `get_structure_with_only_magnetic_atoms`.
   After this, `len(s)` for a structure equals `N_k^mag` — the magnetic-ion count,
   not the total-atom count.

2. **Normalise the energy** (`heisenberg.py:725`):

   ```python
   energies = [e / len(s) for (e, s) in zip(energies, ordered_structures)]
   ```

   This is the single, decisive unit conversion:

   ```
   ε_k = E_k^tot / N_k^mag           [eV]  →  [eV / mag ion]
   ```

   From here on, every energy the mapper touches is in **eV / mag ion**.

3. **De-duplicate and sort** (`heisenberg.py:732-754`). Degenerate orderings are
   dropped using a tolerance of `1e-6` **eV / mag ion**, then structures and
   energies are sorted ascending in `ε_k`.

The cleaned, per-mag-ion energies are stored on the mapper as `self.energies`
(`heisenberg.py:83`). The original total energies are not retained.

---

## Step 2 — Building the Heisenberg design matrix (`_get_exchange_df`)

`_get_exchange_df` (`heisenberg.py:229`) builds one linear equation per ordering.
Each row has:

* an **`E` column** — the left-hand side,
* an **`E0` column** — coefficient of the reference energy,
* one column per neighbour class `ab` — the coefficient of `J_ab`.

**The `E` column (LHS).** `heisenberg.py:330`:

```python
ex_row.loc[sgraph_index, "E"] = self.energies[e_index]
```

so the LHS of row `k` is `ε_k`, in **eV / mag ion**.

**The coupling columns (design coefficients).** For ordering `k`, the code loops
over every magnetic ion `i` and every neighbour `j`, accumulating
`−s_i·s_j` into the column for that bond class (`heisenberg.py:318-321`). Because
the node loop visits each physical bond twice (once from each endpoint), the raw
column equals `−2 Σ s_i s_j`; the explicit ½ factor at `heisenberg.py:338`
(`ex_mat[j_columns] /= 2`) removes the double-count. The resulting coefficient is

```
C_ab,k = − Σ_{⟨i,j⟩ ∈ class ab in cell k}  s_i · s_j         [μ_B²]
```

This coefficient is **extensive**: it is a sum over all bonds of class `ab` in the
supercell, so it scales with `N_k^mag`.

**The `E0` column.** Set to `1` for every row (`heisenberg.py:339`) — a
dimensionless constant carrying the spin-independent reference energy.

So row `k` encodes the equation

```
ε_k  =  E_0 · 1  +  Σ_ab  J_ab · C_ab,k
```

with units, term by term:

```
[eV / mag ion]  =  [eV / mag ion]  +  Σ_ab [J_ab] · [μ_B²]
```

Finally the matrix is made square (`heisenberg.py:342-349`): all-zero columns are
dropped and the rows are truncated so the number of equations equals the number of
unknowns (`E0` plus the retained `J_ab`).

---

## Step 3 — The fit itself (`get_exchange`)

`get_exchange` (`heisenberg.py:353`) solves the square linear system assembled
above. Writing the unknown vector `x = (E_0, J_nn, J_nnn, …)ᵀ`, the design matrix
`H` (the `E0` + coupling columns), and the right-hand side `ε = (ε_1, …, ε_n)ᵀ`:

```
H · x = ε
```

Solved by inversion (`heisenberg.py:376-378`):

```python
H_inv = np.linalg.inv(H)
j_ij  = H_inv @ E          # E here is the ε column, in eV / mag ion
```

**Units coming out of the solve.** `ε` is in eV / mag ion and the coupling columns
`C_ab,k` carry μ_B², so the raw solution components have units

```
[E_0]   = eV / mag ion
[J_ab]  = (eV / mag ion) / μ_B²
```

i.e. each `J_ab` is an exchange energy expressed **per magnetic ion** (because the
LHS was per mag ion while the design coefficients were left extensive — see the
consistency note in §6).

**Conversion to meV.** `heisenberg.py:381`:

```python
j_ij[1:] *= 1000           # eV → meV ; the [0] entry (E_0) is left in eV
```

so the reported couplings are

```
J_ab  in  meV / mag ion         (source labels this "meV/atom")
```

**Degenerate case (fewer than 3 couplings).** If the symmetry analysis yields
fewer than three independent neighbour classes, the matrix route is skipped
(`heisenberg.py:367-373`) and a single averaged `<J>` from §3a is returned instead.

### Step 3a — The `<J>` shortcut (`estimate_exchange`)

`estimate_exchange` (`heisenberg.py:454`) bypasses the matrix and estimates one
average coupling from the lowest FM and AFM orderings, using the same per-mag-ion
energies. With `fm_e`, `afm_e` taken from `self.energies` (`heisenberg.py:468`):

```
Δε      = afm_e − fm_e                     [eV / mag ion]
m_avg   = mean over FM mag ions of |s_i|   [μ_B]
<J>     = Δε / m_avg²                       [eV / (mag ion · μ_B²)]
<J>    *= 1000                              [meV / (mag ion · μ_B²)]   (heisenberg.py:485-487)
```

So `<J>` (stored later as `javg`) is, like the matrix couplings, an exchange energy
**per magnetic ion**, in meV.

---

## Step 4 — What the fitted `HeisenbergModel` carries (`get_heisenberg_model`)

`get_heisenberg_model` (`heisenberg.py:624`) packages the results into the MSONable
`HeisenbergModel` that gets written out. The energy-bearing fields and their units:

| Field (in the written model) | Source | Unit |
|---|---|---|
| `energies` (`hm_energies`, `heisenberg.py:634`) | `self.energies` | **eV / mag ion** — *not* total energies, *not* the original inputs |
| `ex_mat` (`hm_em`, `heisenberg.py:644`) | design matrix, JSON | `E` column eV / mag ion; coupling columns μ_B² |
| `ex_params` (`hm_ep`, `heisenberg.py:645`) | `get_exchange()` | **meV / mag ion** per coupling (or a single `<J>`) |
| `javg` (`hm_javg`, `heisenberg.py:646`) | `estimate_exchange()` | **meV / mag ion** |
| `igraph` (`hm_igraph`, `heisenberg.py:647`) | interaction graph | edge weights = `J_exc` in **meV** (per mag ion); unit label string is `"meV"` |

The structures stored in the model are the **magnetic-only** sublattice cells
produced during cleanup, not the original full cells.

---

## Step 5 — Units summary (end to end)

```
INPUT                    Step 1                      Steps 2–3                 OUTPUT
E_k^tot                  ε_k = E_k^tot / N_k^mag     fit  H·x = ε             ex_params, javg
[eV, whole cell]  ──────▶ [eV / mag ion]      ──────▶ J_ab ×1000      ──────▶ [meV / mag ion]
                          (heisenberg.py:725)         (heisenberg.py:381)      model.energies = ε_k
                                                                               [eV / mag ion]
```

The whole mapper lives in **two** energy units only: **eV / mag ion** (the cleaned
energies and the matrix LHS) and **meV / mag ion** (the reported couplings). Total
atom counts never enter; the only normalisation is by `N_k^mag`.

---

## Step 6 — Consistency caveat (worth flagging)

The fit divides the **left-hand side** by `N_k^mag` (Step 1) but leaves the
**design coefficients** `C_ab,k` extensive (summed over all bonds, Step 2). Putting
those two facts together, the system the code actually solves is

```
E_k^tot / N_k^mag  =  E_0  +  Σ_ab J_ab · C_ab,k .
```

Multiplying back by `N_k^mag` and comparing to the physical total-energy model
`E_k^tot = E_0^tot + Σ_ab J_ab^phys · C_ab,k` shows the two agree, for **all** rows
simultaneously, only if `N_k^mag` is the **same for every ordering**. In that case
the solved couplings equal the physical ones divided by that common magnetic-ion
count — which is exactly why they are "per mag ion." If the supercells differ in
magnetic-ion count, the per-row normalisation no longer corresponds to a single set
of couplings, and the fit is no longer consistent. For orderings enumerated from a
single parent cell (the usual magnetic-orderings workflow) this condition holds; it
is only a hazard if heterogeneous supercell sizes are mixed into one fit.
