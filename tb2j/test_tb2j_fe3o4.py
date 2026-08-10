#!/usr/bin/env python
"""Cross-check the Fe3O4 exchange parameters against TB2J (magnetic force theorem).

A runnable driver, not a unit test: it stages a real VASP + Wannier90 + TB2J
calculation for Fe3O4 and reports the resulting J_ij next to the numbers the
energy-mapping pipeline produced.

Motivation
----------
The energy-mapping fit (results/243) returns, per magnetic ion:

    J_AA (0-0-nn) = -0.753 meV  at 3.66 Ang   (A-A, tetrahedral-tetrahedral)
    J_AB (0-1-nn) = -1.521 meV  at 3.50 Ang   (A-B, the 125 deg superexchange)
    J_BB (1-1-nn) = -1.285 meV  at 2.98 Ang   (B-B, octahedral-octahedral)
    residual      =  6.76,  Vampire Tc = 240 K  (experiment: 858 K)

Fe3O4 is a ferrimagnet whose A-B superexchange dominates by a wide margin;
|J_AA| and |J_BB| should be small fractions of |J_AB|. The fit above makes all
three comparable and carries a large residual. TB2J derives J_ij from a single
reference-state calculation and never solves the (near-singular) linear system,
so it is an independent way to find out whether the discrepancy comes from the
fit *code* or from energy mapping being ill-conditioned on these cells.

Method differences that matter when reading the output
------------------------------------------------------
* Energy mapping fits J's to *large-angle* configuration energies; the magnetic
  force theorem evaluates them as derivatives at one reference state. They are
  different linearisations and are not required to agree exactly.
* TB2J's convention is  E = -sum_{i != j} J_ij e_i . e_j  with UNIT spin
  vectors, summed over ordered pairs. This pipeline's convention
  (notes/heisenberg_energy_units.md) is  E = E0 - sum_<ij> J_ij s_i s_j  with
  s in mu_B, summed over unique bonds, then normalised per magnetic ion.
  ``pmg_to_tb2j_scale()`` spells out the conversion; it is applied only to the
  printed comparison column, never to a check, because that normalisation is
  part of what is under review here.

Stages
------
A  offline; no DFT, no TB2J. Builds the symmetrised Fe3O4 primitive cell,
   classifies the A (2a, tetrahedral) and B (4d, octahedral) Fe sublattices by
   oxygen coordination, sets the ferrimagnetic reference state, and writes the
   VASP SCF + Wannier90 NSCF input sets plus the SLURM script for stage B.
B  the two VASP runs, as a batch job (submit.sh), then wann2J.py over the
   resulting wannier90 files.
C  bins the J_ij into A-A / A-B / B-B shells and checks the Fe3O4 physics
   (A-B antiferromagnetic and dominant), printing the side-by-side comparison.

The DFT is a batch job: nothing in this script ever runs VASP itself, and
stage A / stage bc are both light enough for a login node.

Usage
-----
    .venv/bin/python tb2j/test_tb2j_fe3o4.py --stage a    # write inputs
    sbatch tb2j/tb2j_fe3o4/submit.sh                            # the DFT
    .venv/bin/python tb2j/test_tb2j_fe3o4.py --stage bc   # run TB2J

Exit status is 0 when the physics checks pass, 1 when they fail, and 0 with a
SKIP notice when stage B has nothing to work on yet.

TB2J is NOT a dependency of this project -- installing it into .venv downgrades
scipy (1.18 -> 1.16). Keep it in its own environment and point at it with

    uv venv /path/to/tb2jenv && VIRTUAL_ENV=/path/to/tb2jenv uv pip install TB2J
    export TB2J_PYTHON=/path/to/tb2jenv/bin/python
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

TB2J_DIR = Path(__file__).resolve().parent.parent
REPO = TB2J_DIR.parent

# pymatgen freezes SETTINGS from os.environ at import time, so the POTCAR
# location has to be in place before the imports below. Default to the same
# directory the jobflow-remote workers use (.jfremote/dft_pipeline2.yaml), so
# the script runs from a bare login shell without any module/env setup.
os.environ.setdefault("PMG_VASP_PSP_DIR", str(REPO / ".vasp_psp_compat"))
os.environ.setdefault("PMG_DEFAULT_FUNCTIONAL", "PBE")

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")

from pymatgen.core import Lattice, Structure  # noqa: E402
from pymatgen.io.vasp.inputs import Kpoints  # noqa: E402

# --- Fe3O4 primitive cell, hard-coded so the script needs no network --------
# Inverse spinel, Fd-3m (227). This is the symmetrised primitive cell of the
# relaxed mp-19306 parent structure recorded in results/243/exchange_doc.json
# (conventional a = 8.44397 Ang -> rhombohedral a = 5.970788 Ang, alpha = 60 deg).
# 2 Fe on 2a (tetrahedral, "A"), 4 Fe on 4d (octahedral, "B"), 8 O on 8e.
A_RHOMB = 5.970788
FE_FRAC = [
    [0.000, 0.000, 0.000],
    [0.625, 0.625, 0.125],
    [0.625, 0.625, 0.625],
    [0.250, 0.250, 0.250],
    [0.125, 0.625, 0.625],
    [0.625, 0.125, 0.625],
]
O_FRAC = [
    [0.868650, 0.394051, 0.868650],
    [0.381350, 0.381350, 0.855949],
    [0.381350, 0.381350, 0.381350],
    [0.868650, 0.868650, 0.394051],
    [0.855949, 0.381350, 0.381350],
    [0.868650, 0.868650, 0.868650],
    [0.381350, 0.855949, 0.381350],
    [0.394051, 0.868650, 0.868650],
]

# --- reference state --------------------------------------------------------
# Fe3O4 = Fe3+[A] (Fe3+ Fe2+)[B] O4: the A and B sublattices are antialigned and
# the B sites carry an averaged Fe(2.5+) moment. Net 4 mu_B per formula unit,
# i.e. 8 mu_B in this 2-formula-unit cell -> NUPDOWN = 8.
MAGMOM_A = -5.0
MAGMOM_B = +4.5
NUPDOWN = 8

# --- Wannier setup ----------------------------------------------------------
# VASP's LWANNIER90 interface always overrides num_wann (and num_bands,
# mp_grid, unit_cell_cart, atoms_cart, kpoints) in WANNIER90_WIN with its own
# auto-generated value -- whatever the user writes there is silently ignored.
# The auto value is the full POTCAR valence orbital count: Fe is "d7 s1"
# (s+d, 6 per atom) and O is "s2p4" (s+p, 4 per atom). A projection block
# that omits the s shells (d-only for Fe, p-only for O) undercounts this and
# wannier90 aborts with "param_get_projections: too few projection functions
# defined" -- which VASP's library-mode call then turns into an MPI deadlock
# instead of a clean exit, burning the rest of the walltime.
#   6 Fe x 6 (s+d) + 8 O x 4 (s+p) = 68 Wannier functions per spin channel.
# Both spin channels must be wannierised; TB2J needs _hr.dat for each.
WANN_PROJ_PER_ELEMENT = {"Fe": 6, "O": 4}  # s+d, s+p
EXPECTED_NUM_WANN = 68
KMESH = (5, 5, 5)  # the VASP NSCF mesh AND TB2J's --kmesh
NBANDS = 96        # must exceed num_wann with room to disentangle
DIS_WIN_MAX = 12.0  # eV above E_F, disentanglement window
DIS_FROZ_MAX = 2.0  # eV above E_F, frozen window

# --- standalone wannierisation ------------------------------------------------
# VASP's LWANNIER90 library-mode call (vasp -f Wan90) only computes the overlap
# and projection matrices (wannier90.{1,2}.{amn,mmn,eig}); it never runs the
# actual disentanglement/localisation, so *_hr.dat / *_centres.xyz (what TB2J
# reads) have to come from a separate wannier90.x pass over those files. There
# is no standalone wannier90 module on Justus2, but chem/quantum_espresso/7.1
# bundles a working wannier90.x. VASP writes eigenvalues on an absolute scale,
# not relative to E_F, so dis_win_max/dis_froz_max (meant as "eV above E_F")
# have to be shifted by the actual Fermi energy or wannier90.x's disentangle
# window ends up too narrow ("Energy window contains fewer states than number
# of target WFs"). VASP's spin index 1 is spin-up, 2 is spin-down.
WANN90_MODULE = "chem/quantum_espresso/7.1"
SPIN_SEEDNAMES = {1: "up", 2: "down"}

# --- SLURM resources for stage B --------------------------------------------
# Mirrors the magnetism_justus2 worker in .jfremote/dft_pipeline2.yaml. Account
# bw26b012 is the live one; bw16c005 still shows up in sacctmgr but is retired.
SLURM_PARTITION = "standard"
SLURM_ACCOUNT = "bw26b012"
SLURM_NODES = 1
SLURM_NTASKS_PER_NODE = 24
SLURM_TIME = "06:00:00"  # two VASP runs; the wannierisation is the slow half

# --- energy-mapping reference (results/243), meV per magnetic ion ------------
PMG_REFERENCE = {"AA": -0.7529, "AB": -1.5210, "BB": -1.2851}
PMG_DISTANCES = {"AA": 3.66, "AB": 3.50, "BB": 2.98}
PMG_TC = 240.0
EXPERIMENTAL_TC = 858.0


# ===========================================================================
# Stage A: structure, sublattices, VASP + Wannier90 inputs
# ===========================================================================
def fe3o4_primitive() -> Structure:
    """The hard-coded Fe3O4 primitive cell (6 Fe + 8 O)."""
    lattice = Lattice.from_parameters(A_RHOMB, A_RHOMB, A_RHOMB, 60.0, 60.0, 60.0)
    species = ["Fe"] * len(FE_FRAC) + ["O"] * len(O_FRAC)
    return Structure(lattice, species, FE_FRAC + O_FRAC)


def classify_sublattices(struct: Structure, cutoff: float = 2.4) -> dict[int, str]:
    """Label each Fe site "A" (tetrahedral, 4 O) or "B" (octahedral, 6 O).

    Derived from the oxygen coordination rather than hard-coded indices, so the
    labelling survives any re-ordering of the structure. Fe-O bonds in Fe3O4 are
    ~1.89 Ang (A) and ~2.06 Ang (B), both well inside the 2.4 Ang cutoff, and the
    next shell (Fe-Fe at 2.98 Ang) is well outside it.
    """
    labels: dict[int, str] = {}
    for i, site in enumerate(struct):
        if site.specie.symbol != "Fe":
            continue
        n_oxygen = sum(
            1 for nb in struct.get_neighbors(site, cutoff) if nb.specie.symbol == "O"
        )
        if n_oxygen == 4:
            labels[i] = "A"
        elif n_oxygen == 6:
            labels[i] = "B"
        else:
            raise SystemExit(
                f"Fe site {i} has {n_oxygen} O neighbours within {cutoff} Ang; "
                "expected 4 (tetrahedral A) or 6 (octahedral B)."
            )
    return labels


def reference_magmoms(struct: Structure, labels: dict[int, str]) -> list[float]:
    """Ferrimagnetic reference: A sublattice down, B sublattice up, O zero."""
    return [
        (MAGMOM_A if labels[i] == "A" else MAGMOM_B)
        if site.specie.symbol == "Fe"
        else 0.0
        for i, site in enumerate(struct)
    ]


def num_wann(struct: Structure) -> int:
    """Wannier functions per spin channel for the Fe-d + O-p projection set."""
    return sum(WANN_PROJ_PER_ELEMENT[site.specie.symbol] for site in struct)


def wannier90_win_block(nwann: int) -> str:
    """The WANNIER90_WIN INCAR block (VASP >= 6.2 embeds the .win here).

    VASP fills in kpoints, unit_cell, atoms, mp_grid, num_bands and spinors
    itself; everything else has to be supplied. write_hr/write_xyz produce the
    two files TB2J reads (``*_hr.dat`` and ``*_centres.xyz``).
    """
    return (
        'WANNIER90_WIN = "\n'
        f"num_wann = {nwann}\n"
        "\n"
        "begin projections\n"
        "Fe:s;d\n"
        "O:s;p\n"
        "end projections\n"
        "\n"
        f"dis_win_max   = {DIS_WIN_MAX}\n"
        f"dis_froz_max  = {DIS_FROZ_MAX}\n"
        "dis_num_iter  = 500\n"
        "num_iter      = 500\n"
        "\n"
        "write_hr  = .true.\n"
        "write_xyz = .true.\n"
        "guiding_centres = .true.\n"
        '"\n'
    )


def write_vasp_inputs(struct: Structure, labels: dict[int, str], workdir: Path) -> Path:
    """Write the SCF and Wannier-NSCF VASP input sets under ``workdir``.

    Two directories are produced:
      scf/     ordinary spin-polarised static run (produces CHGCAR/WAVECAR)
      wannier/ non-self-consistent rerun with LWANNIER90 = .TRUE.

    The Wannier step needs the Wannier90-enabled VASP build on the cluster:
    ``vasp -f Wan90`` (VASP 6.4.3 linked against Wannier90 3.1).
    """
    from atomate2.vasp.sets.core import StaticSetGenerator

    struct = struct.copy()
    struct.add_site_property("magmom", reference_magmoms(struct, labels))

    common = {
        "ISPIN": 2,
        "NUPDOWN": NUPDOWN,
        "NELM": 300,
        "ALGO": "Normal",
        "EDIFF": 1e-6,
        "ISMEAR": 0,
        "SIGMA": 0.05,
        "LORBIT": 11,
    }

    scf_dir = workdir / "scf"
    wann_dir = workdir / "wannier"
    scf_dir.mkdir(parents=True, exist_ok=True)
    wann_dir.mkdir(parents=True, exist_ok=True)

    StaticSetGenerator(
        user_incar_settings={**common, "LWAVE": True, "LCHARG": True},
        user_kpoints_settings={"reciprocal_density": 200},
    ).get_input_set(struct, potcar_spec=False).write_input(scf_dir)

    # NSCF: fixed charge density, explicit uniform mesh (Wannier90 needs the full
    # unreduced grid, hence ISYM = -1), plus the Wannier interface.
    StaticSetGenerator(
        user_incar_settings={
            **common,
            "ICHARG": 11,
            "ISYM": -1,
            "LWAVE": False,
            "LCHARG": False,
            "LWANNIER90": True,
            "NBANDS": NBANDS,
        },
        user_kpoints_settings=Kpoints.gamma_automatic(KMESH),
    ).get_input_set(struct, potcar_spec=False).write_input(wann_dir)

    # pymatgen's Incar writer emits one "key = value" line per tag and cannot
    # represent WANNIER90_WIN's quoted multi-line block, so append it as raw text.
    with (wann_dir / "INCAR").open("a") as fh:
        fh.write("\n" + wannier90_win_block(num_wann(struct)))

    write_batch_script(workdir)
    return wann_dir


def write_batch_script(workdir: Path) -> Path:
    """Write the SLURM batch script for stage B.

    Both VASP runs go in one job. The cluster's ``vasp`` wrapper invokes
    srun/mpirun itself from the SBATCH allocation, so it must NOT be prefixed
    with srun/mpirun here (see ``vasp -help``). The Wannier step needs the
    Wannier90-enabled build, hence ``-f Wan90``.

    Resources mirror the magnetism_justus2 worker in .jfremote/dft_pipeline2.yaml,
    with a longer walltime because this is two runs back to back and the
    wannierisation is not cheap.
    """
    script = workdir / "submit.sh"
    script.write_text(
        "#!/bin/bash\n"
        f"#SBATCH --job-name=tb2j_fe3o4\n"
        f"#SBATCH --partition={SLURM_PARTITION}\n"
        f"#SBATCH --account={SLURM_ACCOUNT}\n"
        f"#SBATCH --nodes={SLURM_NODES}\n"
        f"#SBATCH --ntasks-per-node={SLURM_NTASKS_PER_NODE}\n"
        f"#SBATCH --time={SLURM_TIME}\n"
        "#SBATCH --output=%x-%j.out\n"
        "#SBATCH --error=%x-%j.err\n"
        "\n"
        "set -euo pipefail\n"
        "\n"
        "module purge\n"
        "module load chem/vasp/6.4.3 compiler/intel/2024.2.1 mpi/impi/2021.13.1\n"
        "export OMP_NUM_THREADS=1\n"
        "export MKL_NUM_THREADS=1\n"
        "\n"
        f"cd {workdir.resolve()}\n"
        "\n"
        "# 1) spin-polarised SCF -> CHGCAR\n"
        "cd scf\n"
        "vasp\n"
        "cd ..\n"
        "\n"
        "# 2) NSCF wannierisation (ICHARG=11 reads the SCF density)\n"
        "cp scf/CHGCAR wannier/\n"
        "cd wannier\n"
        "vasp -f Wan90\n"
        "cd ..\n"
        "\n"
        "# 3) wannier90.x (both spin channels) + TB2J + the physics check --\n"
        "# stage bc drives all of it, including the standalone wannierisation\n"
        "# VASP's library-mode call doesn't do (see wannierize() in this file).\n"
        f"{sys.executable} {TB2J_DIR / 'tests' / 'test_tb2j_fe3o4.py'} --stage bc "
        f"--workdir {workdir.resolve()}\n"
    )
    script.chmod(0o755)
    return script


# ===========================================================================
# Stage B: run TB2J
# ===========================================================================
def tb2j_python() -> str | None:
    """Interpreter that has TB2J importable, or None if we cannot find one."""
    candidate = os.environ.get("TB2J_PYTHON")
    if candidate and Path(candidate).exists():
        return candidate
    if shutil.which("wann2J.py"):
        return sys.executable
    return None


def wann2j_script(python: str) -> str:
    """Absolute path to wann2J.py belonging to ``python``."""
    local = Path(python).parent / "wann2J.py"
    if local.exists():
        return str(local)
    found = shutil.which("wann2J.py")
    if found:
        return found
    raise SystemExit(f"wann2J.py not found next to {python} or on PATH")


def wannier_outputs_present(wann_dir: Path) -> bool:
    """True when both spin channels have the two files TB2J needs."""
    return all(
        (wann_dir / f"wannier90.{spin}{suffix}").exists()
        for spin in ("up", "down")
        for suffix in ("_hr.dat", "_centres.xyz")
    )


def fermi_energy(wann_dir: Path) -> float:
    """Read E_F from the NSCF vasprun.xml."""
    from pymatgen.io.vasp.outputs import Vasprun

    return float(Vasprun(str(wann_dir / "vasprun.xml")).efermi)


def wannier_overlaps_present(wann_dir: Path) -> bool:
    """True when VASP's LWANNIER90 step has produced the raw overlap files."""
    return (wann_dir / "wannier90.win").exists() and all(
        (wann_dir / f"wannier90.{spin}.{ext}").exists()
        for spin in SPIN_SEEDNAMES
        for ext in ("amn", "mmn", "eig")
    )


def spin_win_header(efermi: float) -> str:
    """Standalone wannier90.x .win header with E_F-shifted disentangle windows.

    VASP eigenvalues (and hence dis_win_max/dis_froz_max) are absolute, not
    relative to E_F, so the shift has to be applied here where E_F is finally
    known.
    """
    return (
        "begin projections\n"
        "Fe:s;d\n"
        "O:s;p\n"
        "end projections\n"
        "\n"
        f"fermi_energy  = {efermi:.6f}\n"
        f"dis_win_max   = {efermi + DIS_WIN_MAX:.6f}\n"
        f"dis_froz_max  = {efermi + DIS_FROZ_MAX:.6f}\n"
        "dis_num_iter  = 500\n"
        "num_iter      = 500\n"
        "\n"
        "write_hr  = .true.\n"
        "write_xyz = .true.\n"
        "guiding_centres = .true.\n"
    )


def wannierize(wann_dir: Path, efermi: float) -> None:
    """Run standalone wannier90.x for both spin channels.

    Reuses the unit_cell/atoms/kpoints block VASP auto-generated in
    wannier90.win (everything from the "generated automatically by VASP"
    marker on) so num_bands/mp_grid/kpoints stay exactly what the AMN/MMN/EIG
    files were computed with.
    """
    marker = "# This part was generated automatically by VASP"
    shared = (wann_dir / "wannier90.win").read_text()
    auto_block = shared[shared.index(marker):]

    for spin, name in SPIN_SEEDNAMES.items():
        seed = f"wannier90.{name}"
        (wann_dir / f"{seed}.win").write_text(spin_win_header(efermi) + "\n" + auto_block)
        for ext in ("amn", "mmn", "eig"):
            shutil.copy(wann_dir / f"wannier90.{spin}.{ext}", wann_dir / f"{seed}.{ext}")
        subprocess.run(
            ["bash", "-lc",
             f"module purge && module load {WANN90_MODULE} && "
             f"srun -n 1 wannier90.x {seed}"],
            check=True, cwd=wann_dir,
        )


def run_tb2j(wann_dir: Path, efermi: float, python: str) -> Path:
    """Invoke wann2J.py; return the TB2J_results directory."""
    out = wann_dir / "TB2J_results"
    cmd = [
        python, wann2j_script(python),
        "--path", str(wann_dir),
        "--posfile", "POSCAR",
        "--prefix_up", "wannier90.up",
        "--prefix_down", "wannier90.down",
        "--elements", "Fe",
        "--efermi", f"{efermi}",
        "--kmesh", *[str(k) for k in KMESH],
        "--emin", "-14.0",
        "--emax", "0.0",
        "--output_path", str(out),
    ]
    print("  $ " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=wann_dir)
    return out


# ===========================================================================
# Stage C: parse and check
# ===========================================================================
# Runs inside the TB2J interpreter so TB2J never has to be importable from .venv.
_PARSE_SCRIPT = r"""
import json, sys
import numpy as np
from TB2J.io_exchange import SpinIO

obj = SpinIO.load_pickle(path=sys.argv[1])
labels = json.loads(sys.argv[2])            # {"atom index": "A"/"B"}
out = {}
for (R, i, j), jval in obj.exchange_Jdict.items():
    la, lb = labels[str(obj.iatom(i))], labels[str(obj.iatom(j))]
    pair = "".join(sorted(la + lb))
    dist = round(float(obj.distance_dict[(R, i, j)][1]), 2)  # (vector, distance)
    out.setdefault("%s@%.2f" % (pair, dist), []).append(float(jval) * 1e3)  # eV -> meV
json.dump({k: [float(np.mean(v)), len(v)] for k, v in out.items()}, sys.stdout)
"""


def load_tb2j(results_dir: Path, labels: dict[int, str], python: str) -> dict:
    """Read TB2J's pickled SpinIO and reduce it to per-shell J's in meV.

    TB2J stores ``exchange_Jdict`` as {(R, i, j): J} in **eV**, where i/j are
    *spin* indices (magnetic atoms only) and R is the lattice translation. Every
    pair is binned by (sublattice pair, rounded distance) and averaged within a
    bin, giving the same AA / AB / BB shell language as the energy-mapping fit.
    """
    proc = subprocess.run(
        [python, "-c", _PARSE_SCRIPT, str(results_dir),
         json.dumps({str(k): v for k, v in labels.items()})],
        check=True, capture_output=True, text=True,
    )
    return json.loads(proc.stdout)


def nearest_shells(shells: dict) -> dict[str, tuple[float, float]]:
    """For each of AA/AB/BB pick the nearest shell. Returns {pair: (J_meV, d)}."""
    best: dict[str, tuple[float, float]] = {}
    for key, (jval, _count) in shells.items():
        pair, dist = key.split("@")
        dist = float(dist)
        if pair not in best or dist < best[pair][1]:
            best[pair] = (jval, dist)
    return best


def pmg_to_tb2j_scale(pair: str) -> float:
    """Factor converting a pipeline J (meV / magnetic ion) to TB2J's convention.

    Pipeline (notes/heisenberg_energy_units.md):
        E^tot = E0 - sum_<ij> J^pmg_ij s_i s_j,   s in mu_B, unique bonds
    TB2J:
        E     =    - sum_{i != j} J^tb_ij e_i . e_j,  unit vectors, ordered pairs

    A sum over ordered pairs is twice a sum over unique bonds, so
        J^tb_ij = (1/2) J^pmg_ij * s_i * s_j.
    Printed only -- see the module docstring.
    """
    s = {"A": abs(MAGMOM_A), "B": abs(MAGMOM_B)}
    return 0.5 * s[pair[0]] * s[pair[1]]


# ===========================================================================
# Drivers
# ===========================================================================
def stage_a(workdir: Path) -> dict[int, str]:
    print("=== Stage A: Fe3O4 cell, sublattices, VASP + Wannier90 inputs ===")
    struct = fe3o4_primitive()
    print(f"  formula {struct.composition.reduced_formula}, {len(struct)} sites")
    print(f"  lattice abc={[round(x, 4) for x in struct.lattice.abc]} "
          f"angles={[round(x, 2) for x in struct.lattice.angles]}")

    labels = classify_sublattices(struct)
    n_a = sum(1 for v in labels.values() if v == "A")
    n_b = sum(1 for v in labels.values() if v == "B")
    print(f"  Fe sublattices by O coordination: {n_a} x A (2a, tetrahedral), "
          f"{n_b} x B (4d, octahedral)")
    if (n_a, n_b) != (2, 4):
        raise SystemExit(f"expected 2 A-site and 4 B-site Fe, got {n_a} and {n_b}")

    net = sum(reference_magmoms(struct, labels))
    print(f"  ferrimagnetic reference: A={MAGMOM_A:+.1f}, B={MAGMOM_B:+.1f} mu_B, "
          f"net {net:+.1f} mu_B/cell ({net / 2:+.1f} mu_B per formula unit)")
    if not np.isclose(net, NUPDOWN):
        raise SystemExit(f"net moment {net} does not match NUPDOWN={NUPDOWN}")

    nwann = num_wann(struct)
    print(f"  Wannier functions per spin channel: 6 Fe x 6 (s+d) + 8 O x 4 (s+p) = {nwann}")
    if nwann != EXPECTED_NUM_WANN:
        raise SystemExit(f"expected {EXPECTED_NUM_WANN} Wannier functions, got {nwann}")
    if nwann >= NBANDS:
        raise SystemExit(f"NBANDS={NBANDS} must exceed num_wann={nwann}")

    wann_dir = write_vasp_inputs(struct, labels, workdir)
    incar = (wann_dir / "INCAR").read_text()
    for needle in ("LWANNIER90 = True", "write_hr  = .true.", f"num_wann = {nwann}"):
        if needle not in incar:
            raise SystemExit(f"Wannier INCAR is missing {needle!r}")

    print(f"  wrote {workdir}/scf and {workdir}/wannier")
    print(f"  next:  sbatch {workdir}/submit.sh   "
          f"({SLURM_NODES}x{SLURM_NTASKS_PER_NODE} tasks, {SLURM_TIME}, "
          f"account {SLURM_ACCOUNT})")
    print("  -> Stage A OK\n")
    return labels


def stages_bc(workdir: Path, labels: dict[int, str]) -> bool | None:
    """Run TB2J and check the Fe3O4 physics. Returns None when skipped."""
    wann_dir = workdir / "wannier"
    python = tb2j_python()

    print("=== Stage B: TB2J (magnetic force theorem) ===")
    if python is None:
        print("  SKIP: TB2J not found. Install it into its own venv and set "
              "TB2J_PYTHON=<venv>/bin/python")
        return None
    print(f"  TB2J interpreter: {python}")
    if not wannier_outputs_present(wann_dir):
        if not wannier_overlaps_present(wann_dir):
            print(f"  SKIP: no wannier90 overlap files in {wann_dir}")
            print(f"        submit stage B first: sbatch {workdir}/submit.sh")
            return None
        efermi = fermi_energy(wann_dir)
        print(f"  wannierising both spin channels (E_F = {efermi:.4f} eV)...")
        wannierize(wann_dir, efermi)
        if not wannier_outputs_present(wann_dir):
            raise SystemExit(
                "wannier90.x did not produce *_hr.dat/_centres.xyz -- "
                f"check {wann_dir}/wannier90.{{up,down}}.wout"
            )

    results_dir = wann_dir / "TB2J_results"
    if (results_dir / "TB2J.pickle").exists():
        print(f"  reusing existing {results_dir}")
    else:
        efermi = fermi_energy(wann_dir)
        print(f"  E_F = {efermi:.4f} eV (from the NSCF vasprun.xml)")
        results_dir = run_tb2j(wann_dir, efermi, python)

    print("\n=== Stage C: shell-resolved J_ij and the Fe3O4 physics check ===")
    shells = load_tb2j(results_dir, labels, python)
    print(f"  {'shell':<12} {'J [meV]':>10} {'n pairs':>8}")
    for key in sorted(shells, key=lambda k: (k.split("@")[0], float(k.split("@")[1]))):
        jval, count = shells[key]
        print(f"  {key:<12} {jval:>10.4f} {count:>8}")

    best = nearest_shells(shells)
    missing = {"AA", "AB", "BB"} - set(best)
    if missing:
        raise SystemExit(f"TB2J produced no {sorted(missing)} pairs -- check --rcut/kmesh")

    print(f"\n  {'pair':<6} {'d [Ang]':>8} {'TB2J [meV]':>12} "
          f"{'energy-map [meV]':>18} {'energy-map rescaled':>21}")
    for pair in ("AA", "AB", "BB"):
        jtb, dist = best[pair]
        jpmg = PMG_REFERENCE[pair]
        print(f"  {pair:<6} {dist:>8.2f} {jtb:>12.4f} {jpmg:>18.4f} "
              f"{jpmg * pmg_to_tb2j_scale(pair):>21.4f}")
    print("  (energy-map distances were "
          + ", ".join(f"{p} {PMG_DISTANCES[p]:.2f}" for p in ("AA", "AB", "BB"))
          + " Ang)")
    print("  (the rescaled column applies pmg_to_tb2j_scale; diagnostic only,")
    print("   it is NOT one of the checks below -- see the module docstring)")

    j_aa, j_ab, j_bb = (best[p][0] for p in ("AA", "AB", "BB"))
    checks = {
        "A-B superexchange is antiferromagnetic (J_AB < 0)": j_ab < 0,
        "A-B dominates A-A (|J_AB| > |J_AA|)": abs(j_ab) > abs(j_aa),
        "A-B dominates B-B (|J_AB| > |J_BB|)": abs(j_ab) > abs(j_bb),
        "A-A is a minor coupling (|J_AA| < 0.5 |J_AB|)": abs(j_aa) < 0.5 * abs(j_ab),
    }
    print()
    for label, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")

    # For context: the energy-mapping fit scrapes past these same checks
    # (|J_AA|/|J_AB| = 0.49, just inside the 0.5 bound). Its problem is not the
    # ordering of the couplings but their magnitudes -- hence the Tc line.
    print(f"\n  energy-mapping fit for comparison: "
          f"|J_AA|/|J_AB| = {abs(PMG_REFERENCE['AA'] / PMG_REFERENCE['AB']):.2f}, "
          f"|J_BB|/|J_AB| = {abs(PMG_REFERENCE['BB'] / PMG_REFERENCE['AB']):.2f},")
    print(f"  Tc = {PMG_TC:.0f} K against {EXPERIMENTAL_TC:.0f} K measured "
          f"(ratio {PMG_TC / EXPERIMENTAL_TC:.2f}).")
    jtb_ab = best["AB"][0]
    print(f"  TB2J here: |J_AA|/|J_AB| = {abs(j_aa / jtb_ab):.2f}, "
          f"|J_BB|/|J_AB| = {abs(j_bb / jtb_ab):.2f}.")
    ok = all(checks.values())
    print("  ->", "PASS" if ok else "FAIL")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--workdir", type=Path, default=TB2J_DIR / "tb2j_fe3o4",
        help="directory for the staged VASP/Wannier calculation "
             "(default: %(default)s)",
    )
    parser.add_argument(
        "--stage", choices=["a", "bc"], default="a",
        help="'a' writes the VASP inputs and the SLURM script; 'bc' runs TB2J "
             "on the finished calculation (default: %(default)s)",
    )
    args = parser.parse_args()

    if args.stage == "a":
        args.workdir.mkdir(parents=True, exist_ok=True)
        stage_a(args.workdir)
        print(f"Stage A done. Next: sbatch {args.workdir}/submit.sh")
        return 0

    # Stage bc: do not touch the existing inputs, just re-derive the labels.
    if not (args.workdir / "wannier").is_dir():
        raise SystemExit(
            f"{args.workdir} has no wannier/ directory -- run --stage a first."
        )
    labels = classify_sublattices(fe3o4_primitive())
    result = stages_bc(args.workdir, labels)
    return 0 if result is not False else 1


if __name__ == "__main__":
    sys.exit(main())
