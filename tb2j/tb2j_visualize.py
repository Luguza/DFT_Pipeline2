"""Visualize the results of a raw TB2J + Vampire run.

Usage: python tb2j_visualize.py <TB2J_results_dir> [--cutoff 3.6]

Unlike ``visualize_results.py`` (which reads an atomate2 ``ExchangeDocument``
fitted across several magnetic orderings via a jobflow-remote job store), this
reads directly from a ``TB2J_results`` directory produced by ``wann2J.py``:
``TB2J.pickle`` for the raw per-bond exchange parameters and ``Vampire/`` for
the Monte Carlo output. There is only one magnetic ordering here (the DFT
calculation's own), so there is no ordering-energies plot, and this run's
``input`` only requests ``output:material-magnetisation`` (no
``mean-susceptibility``), so there is no susceptibility plot either.

Plots are written to ``<TB2J_results_dir>/plots/``.
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

SERIES_COLORS = [
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
]
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"


def load_tb2j_pickle(tb2j_dir: Path) -> dict:
    with open(tb2j_dir / "TB2J.pickle", "rb") as f:
        return pickle.load(f)


def load_vampire_materials(tb2j_dir: Path) -> dict[int, tuple[str, str, float]]:
    """Parse ``Vampire/vampire.mat`` into ``{material_id: (element, "up"|"down", moment)}``."""
    text = (tb2j_dir / "Vampire" / "vampire.mat").read_text()
    materials: dict[int, tuple[str, str, float]] = {}
    for mat_id_str in re.findall(r"material\[(\d+)\]:material-element", text):
        mat_id = int(mat_id_str)
        element = re.search(rf"material\[{mat_id}\]:material-element=(\S+)", text)[1]
        moment = float(
            re.search(rf"material\[{mat_id}\]:atomic-spin-moment=([-\d.eE]+)", text)[1]
        )
        spin_dir = re.search(
            rf"material\[{mat_id}\]:initial-spin-direction\s*=\s*([-\d.,]+)", text
        )[1]
        z = float(spin_dir.split(",")[-1])
        materials[mat_id] = (element, "up" if z > 0 else "down", moment)
    return materials


def load_vampire_output(tb2j_dir: Path, n_mats: int) -> pd.DataFrame:
    """Parse ``Vampire/output`` (columns: T, then mx,my,mz,m_length per material)."""
    names = ["T"]
    for i in range(n_mats):
        names += [f"mx_{i + 1}", f"my_{i + 1}", f"mz_{i + 1}", f"m_{i + 1}"]
    return pd.read_csv(
        tb2j_dir / "Vampire" / "output", sep=r"\s+", comment="#", header=None, names=names
    )


def plot_magnetization(tb2j_dir: Path, outdir: Path) -> None:
    """Plot the Vampire M(T) curves, weighted by each material's moment.

    The run's ``input`` only requested ``output:material-magnetisation``
    (mx, my, mz, |m| per material), not the mean-magnetisation-length VAMPIRE
    would otherwise report directly. The total is reconstructed from the
    per-material magnetization *vectors* (weighted by atomic spin moment,
    assuming equal atom counts per material, true here since every material
    is exactly one site of the home cell) so antiparallel sublattices cancel
    the way they physically should in a ferrimagnet like Fe3O4.
    """
    materials = load_vampire_materials(tb2j_dir)
    n_mats = len(materials)
    df = load_vampire_output(tb2j_dir, n_mats)

    weights = np.array([materials[i + 1][2] for i in range(n_mats)])
    mx = df[[f"mx_{i + 1}" for i in range(n_mats)]].to_numpy()
    my = df[[f"my_{i + 1}" for i in range(n_mats)]].to_numpy()
    mz = df[[f"mz_{i + 1}" for i in range(n_mats)]].to_numpy()
    m_len = df[[f"m_{i + 1}" for i in range(n_mats)]].to_numpy()
    vec = np.stack([mx, my, mz], axis=-1) * m_len[..., None]
    total_vec = (vec * weights[None, :, None]).sum(axis=1) / weights.sum()
    m_total = np.linalg.norm(total_vec, axis=-1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["T"], m_total, color=SERIES_COLORS[0], lw=2, label="total")
    for i in range(n_mats):
        element, spin, m_s = materials[i + 1]
        arrow = r"$\uparrow$" if spin == "up" else r"$\downarrow$"
        label = f"{element} {arrow} ($M_s$ = {m_s:.2f} $\\mu_B$)"
        ax.plot(
            df["T"],
            df[f"m_{i + 1}"],
            color=SERIES_COLORS[(i + 1) % len(SERIES_COLORS)],
            lw=2,
            alpha=0.5,
            label=label,
        )

    # Tc as the steepest drop of the total curve; only meaningful if the run
    # actually captured the transition (total magnetization decays well below 1).
    if m_total.min() < 0.5:
        dmdt = np.gradient(m_total, df["T"])
        tc = df["T"].iloc[np.argmin(dmdt)]
        ax.axvline(tc, color=INK_MUTED, ls="--", lw=1.5, label=f"$T_c$ (est.) = {tc:.0f} K")

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"$M / M_\mathrm{sat}$")
    ax.set_title("TB2J + VAMPIRE Monte Carlo")
    ax.grid(color=GRIDLINE, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()

    out = outdir / "magnetization.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_exchange_graph(tb2j_dir: Path, outdir: Path, cutoff: float) -> None:
    """Draw the interaction graph of the home-cell magnetic sites.

    Nodes are the magnetic sites of ``TB2J.pickle`` (``index_spin >= 0``),
    coloured by spin direction. Edges are built directly from
    ``exchange_Jdict``/``distance_dict`` (the raw per-bond J_iso TB2J fit to
    the DFT Hamiltonian, in meV) rather than a shell-averaged Heisenberg fit -
    there is only one magnetic ordering here, so no such fit exists. Distinct
    periodic images of the same (site, site) pair at the same distance are
    averaged into a single edge, annotated with the mean J and the image
    count; edges are only kept up to ``cutoff`` Angstrom to keep the graph
    readable (site pairs beyond it are still in ``exchange.out``/``JvsR.pdf``).
    """
    obj = load_tb2j_pickle(tb2j_dir)
    atoms = obj["atoms"]
    index_spin = obj["index_spin"]
    spinat = obj["spinat"]
    jdict = obj["exchange_Jdict"]
    ddict = obj["distance_dict"]

    spin_to_atom = {s: i for i, s in enumerate(index_spin) if s >= 0}
    n_spins = len(spin_to_atom)

    groups: dict[tuple[int, int, float], list[float]] = {}
    for key, j_eV in jdict.items():
        R, si, sj = key
        if si >= sj:
            continue
        dist = ddict[key][1]
        if dist > cutoff or dist < 1e-6:
            continue
        shell = round(dist, 2)
        groups.setdefault((si, sj, shell), []).append(j_eV * 1000)

    shells = sorted({shell for _, _, shell in groups})
    shell_colors = {d: SERIES_COLORS[(i + 1) % len(SERIES_COLORS)] for i, d in enumerate(shells)}

    edges = [(si, sj, shell, np.mean(vals), len(vals)) for (si, sj, shell), vals in groups.items()]
    edges.sort(key=lambda e: (e[2], e[0], e[1]))

    nodes = list(range(n_spins))
    pos = nx.circular_layout(nodes) if len(nodes) > 1 else {nodes[0]: np.zeros(2)}
    pos = {n: np.asarray(p, dtype=float) for n, p in pos.items()}

    fig, ax = plt.subplots(figsize=(7, 5.5))

    def edge_label(xy: np.ndarray, weight: float, mult: int, color: str) -> None:
        text = f"{weight:+.1f}" + (f" (×{mult})" if mult > 1 else "")
        ax.annotate(text, xy, color=color, ha="center", va="center",
                    fontsize=7.5, fontweight="bold", zorder=5,
                    bbox={"boxstyle": "round,pad=0.15", "fc": "white",
                          "ec": "none", "alpha": 0.85})

    RAD = 0.16
    for node_pair in {(u, v) for u, v, *_ in edges}:
        group = [e for e in edges if (e[0], e[1]) == node_pair]
        m = len(group)
        for k, (u, v, shell, weight, mult) in enumerate(group):
            rad = RAD + 0.35 * (k - (m - 1) / 2)
            color = shell_colors[shell]
            ax.add_patch(FancyArrowPatch(
                pos[u], pos[v], connectionstyle=f"arc3,rad={rad}", arrowstyle="-",
                color=color, lw=2.5, shrinkA=30, shrinkB=30, zorder=2,
            ))
            delta = pos[v] - pos[u]
            mid = (pos[u] + pos[v]) / 2 + 0.5 * rad * np.array([delta[1], -delta[0]])
            edge_label(mid, weight, mult, color)

    for n in nodes:
        atom_idx = spin_to_atom[n]
        up = spinat[atom_idx][2] > 0
        ax.scatter(*pos[n], s=3200, zorder=3,
                   color=SERIES_COLORS[0] if up else SERIES_COLORS[5],
                   edgecolors="white", linewidths=2)
        moment = np.linalg.norm(spinat[atom_idx])
        ax.annotate(
            rf"$\mathrm{{{atoms.get_chemical_symbols()[atom_idx]}}}_{{{n}}}$"
            "\n"
            rf"${moment:.2f}\,\mu_B$",
            pos[n], color="white", ha="center", va="center",
            fontsize=9, fontweight="bold", zorder=4, linespacing=1.4,
        )

    handles = [
        Line2D([0], [0], color=shell_colors[d], lw=2.5, label=f"$d$ = {d:.2f} $\\AA$")
        for d in shells
    ]
    if handles:
        ax.legend(handles=handles, frameon=False, loc="upper center",
                  bbox_to_anchor=(0.5, 0.0), ncol=1, fontsize=9,
                  handlelength=1.6, borderaxespad=0.0,
                  title=r"edge labels: $J_\mathrm{iso}$ in meV (TB2J)",
                  title_fontsize=8)

    ax.set_title(f"{atoms.get_chemical_formula()} exchange couplings (d < {cutoff:g} Å)", pad=2)
    ax.set_aspect("equal")
    ax.margins(0.18)
    ax.axis("off")
    fig.tight_layout()

    out = outdir / "exchange_graph.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tb2j_dir", type=Path, help="Path to a TB2J_results directory")
    parser.add_argument(
        "--cutoff", type=float, default=3.6,
        help="Max bond distance (Angstrom) shown in the exchange graph (default: 3.6)",
    )
    args = parser.parse_args()

    tb2j_dir = args.tb2j_dir.resolve()
    if not (tb2j_dir / "TB2J.pickle").exists():
        raise SystemExit(f"{tb2j_dir} has no TB2J.pickle - is this a TB2J_results directory?")

    outdir = tb2j_dir / "plots"
    outdir.mkdir(exist_ok=True)

    if (tb2j_dir / "Vampire" / "output").exists():
        plot_magnetization(tb2j_dir, outdir)
    else:
        print("No Vampire/output found; skipping magnetization plot.")
    plot_exchange_graph(tb2j_dir, outdir, args.cutoff)


if __name__ == "__main__":
    main()
