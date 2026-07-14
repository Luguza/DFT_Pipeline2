"""Visualize the results of an exchange workflow.

Usage: python visualize_results.py <db_id>

``db_id`` is the jobflow-remote db_id of any job in the exchange flow (as shown
by ``jf job list``). Plots are written to ``results/<db_id>/``.

Each ``plot_*`` function produces one figure and is independent of the others,
so they can later be toggled individually via CLI flags.
"""

from __future__ import annotations

import argparse
from io import StringIO
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from jobflow_remote import JobController
from matplotlib.patches import FancyArrowPatch
from monty.json import MontyDecoder

PROJECT = "dft_pipeline2"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

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


def load_exchange_doc(db_id: str) -> dict:
    """Load the raw ExchangeDocument dict from the flow containing job ``db_id``."""
    jc = JobController.from_project_name(PROJECT)
    flows = jc.get_flows_info(db_ids=[db_id], with_jobs_info=True)
    if not flows:
        raise SystemExit(f"No flow contains a job with db_id {db_id}.")
    flow = flows[0]
    try:
        idx = flow.job_names.index("build exchange doc")
    except ValueError:
        raise SystemExit(
            f"Flow '{flow.name}' ({flow.flow_id}) has no 'build exchange doc' job "
            "- is this an exchange flow?"
        ) from None
    raw = jc.get_job_output(db_id=flow.db_ids[idx], load=True)
    if raw is None:
        raise SystemExit(
            f"Job '{flow.job_names[idx]}' (db_id {flow.db_ids[idx]}) has no output "
            "yet - has the flow finished?"
        )
    return raw, flow.db_ids


def material_moments(doc: dict) -> dict[int, tuple[str, str, float]]:
    """Per-material saturation moments M_sat that Vampire normalizes to.

    Vampire outputs the mean magnetization *length*, i.e. M/M_sat, where each
    material's M_sat is its atomic spin moment. This mirrors atomate2's
    ``VampireCaller._create_mat``: one material per (sublattice, spin sign)
    group, with the moment taken as ``|magmom|`` of a representative site.

    Returns ``{material_id: (element, "up"|"down", M_sat_in_muB)}``.
    """
    hm = doc.get("heisenberg_model")
    if not hm:
        return {}
    structure = MontyDecoder().process_decoded(hm["structures"][0])
    site_labels = hm["site_labels"][0]
    magmoms = structure.site_properties["magmom"]

    reps: dict[int, int] = {}  # material id (1-indexed) -> representative site
    mat_ids: dict[tuple[int, bool], int] = {}
    for site, (sub_id, magmom) in enumerate(zip(site_labels, magmoms, strict=True)):
        group = (sub_id, magmom > 0)
        mat_ids.setdefault(group, len(mat_ids) + 1)
        reps.setdefault(mat_ids[group], site)

    return {
        mat_id: (
            structure[site].specie.symbol,
            "up" if magmoms[site] > 0 else "down",
            abs(magmoms[site]),
        )
        for mat_id, site in sorted(reps.items())
    }


def plot_magnetization(doc: dict, outdir: Path, save_df=False) -> None:
    """Plot the Vampire M(T) curves with the critical temperature marked."""
    if not doc.get("vampire_output"):
        print("Exchange doc has no vampire output; skipping magnetization plot.")
        return

    vout = MontyDecoder().process_decoded(doc["vampire_output"])
    df = pd.read_json(StringIO(vout.parsed_out))
    if save_df:
        out = outdir / "magnetization.csv"
        df.to_csv(out, index=False)
        print(f"Wrote {out}")
    tc = vout.critical_temp
    moments = material_moments(doc)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["T"], df["m_total"], color=SERIES_COLORS[0], lw=2, label="total")
    for i in range(vout.nmats):
        mat_id = i + 1
        if mat_id in moments:
            element, spin, m_s = moments[mat_id]
            arrow = r"$\uparrow$" if spin == "up" else r"$\downarrow$"
            label = f"{element} {arrow} ($M_s$ = {m_s:.2f} $\\mu_B$)"
        else:
            label = f"material {mat_id}"
        ax.plot(
            df["T"],
            df[f"m_{mat_id}"],
            color=SERIES_COLORS[(i + 1) % len(SERIES_COLORS)],
            # marker=".",
            lw=2,
            alpha=0.5,
            label=label,
        )
    ax.axvline(tc, color=INK_MUTED, ls="--", lw=1.5, label=f"$T_c$ = {tc:.0f} K")

    for mat_id, (element, spin, m_s) in moments.items():
        print(f"M_sat (material {mat_id}, {element} {spin}) = {m_s:.2f} muB")

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"$M / M_\mathrm{sat}$")
    ax.set_title(f"{doc.get('formula_pretty', '')} Vampire Monte Carlo".strip())
    ax.grid(color=GRIDLINE, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()

    out = outdir / "magnetization.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Wrote {out}")

def plot_exchange_graph(doc: dict, outdir: Path) -> None:
    """Draw the interaction graph: unique magnetic sites joined by J couplings.

    Nodes are the symmetry-distinct magnetic sites of the ground-state
    interaction graph (``HeisenbergModel.igraph``); edges are the fitted
    exchange constants, one per neighbour shell (nn, nnn, ...), annotated with
    ``J`` (meV) and the bond distance. Intra-site couplings are self-loops.
    """
    hm = doc.get("heisenberg_model")
    if not hm:
        print("Exchange doc has no heisenberg model; skipping exchange graph.")
        return

    ig = MontyDecoder().process_decoded(hm["igraph"])
    graph = ig.graph
    structure = ig.structure
    magmoms = structure.site_properties["magmom"]

    # Map each fitted J value back to its shell name (nn/nnn/...) and distance.
    dists = hm.get("dists", {})
    weight_to_shell: dict[float, tuple[str, float | None]] = {}
    for key, j in hm.get("ex_params", {}).items():
        if key == "E0":
            continue
        shell = key.split("-")[-1]
        weight_to_shell[round(j, 8)] = (shell, dists.get(shell))
    shell_order = list(dists) or ["nn", "nnn", "nnnn"]

    # Collapse the many periodic-image edges into one per (site pair, shell).
    seen: set[tuple[int, int, float]] = set()
    edges: list[tuple[int, int, float, str, float | None]] = []
    for u, v, data in graph.edges(data=True):
        j = round(data["weight"], 8)
        dedup = (min(u, v), max(u, v), j)
        if dedup in seen:
            continue
        seen.add(dedup)
        shell, dist = weight_to_shell.get(j, ("?", None))
        edges.append((u, v, j, shell, dist))

    nodes = list(graph.nodes())
    pos = nx.circular_layout(nodes) if len(nodes) > 1 else {nodes[0]: np.zeros(2)}
    pos = {n: np.asarray(p, dtype=float) for n, p in pos.items()}
    centroid = np.mean(list(pos.values()), axis=0)

    shell_colors = {
        sh: SERIES_COLORS[(i + 1) % len(SERIES_COLORS)]
        for i, sh in enumerate(shell_order)
    }

    fig, ax = plt.subplots(figsize=(8, 4))

    # Nodes: coloured by spin direction, labelled with element + arrow.
    for n in nodes:
        up = magmoms[n] > 0
        ax.scatter(*pos[n], s=1600, zorder=3,
                   color=SERIES_COLORS[0] if up else SERIES_COLORS[5],
                   edgecolors="white", linewidths=2)
        arrow = r"$\uparrow$" if up else r"$\downarrow$"
        ax.annotate(f"{structure[n].specie.symbol}{arrow}", pos[n],
                    color="white", ha="center", va="center",
                    fontsize=13, fontweight="bold", zorder=4)

    # Edges: fan out parallel shells between a pair; self-loops for intra-site.
    inter = [e for e in edges if e[0] != e[1]]
    for pair in {(min(u, v), max(u, v)) for u, v, *_ in inter}:
        group = [e for e in inter if (min(e[0], e[1]), max(e[0], e[1])) == pair]
        group.sort(key=lambda e: shell_order.index(e[3]) if e[3] in shell_order else 99)
        m = len(group)
        for k, (u, v, j, shell, dist) in enumerate(group):
            rad = 0.35 * (k - (m - 1) / 2)
            a, b = pos[u], pos[v]
            ax.add_patch(FancyArrowPatch(
                a, b, connectionstyle=f"arc3,rad={rad}", arrowstyle="-",
                color=shell_colors.get(shell, INK_MUTED), lw=2.5,
                shrinkA=22, shrinkB=22, zorder=2,
            ))
            mid = (a + b) / 2
            perp = np.array([b[1] - a[1], a[0] - b[0]])
            lbl = mid + 0.5 * rad * perp
            _edge_label(ax, lbl, shell, j, dist, shell_colors.get(shell, INK_MUTED))

    for k, (u, _v, j, shell, dist) in enumerate(e for e in edges if e[0] == e[1]):
        direction = pos[u] - centroid
        if np.linalg.norm(direction) < 1e-9:
            direction = np.array([np.cos(k), np.sin(k)])
        direction = direction / np.linalg.norm(direction)
        r = 0.22
        loop = pos[u] + direction * r
        ax.add_patch(plt.Circle(loop, r, fill=False, zorder=2, lw=2.5,
                                color=shell_colors.get(shell, INK_MUTED)))
        _edge_label(ax, pos[u] + direction * (2 * r + 0.12), shell, j, dist,
                    shell_colors.get(shell, INK_MUTED))

    ax.set_title(f"{doc.get('formula_pretty', '')} exchange couplings".strip(), pad=2)
    ax.set_aspect("equal")
    ax.margins(0.18)
    ax.axis("off")
    fig.tight_layout()

    out = outdir / "exchange_graph.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Wrote {out}")


def _edge_label(ax, xy, shell, j, dist, color) -> None:
    """Annotate an exchange edge with its shell, J (meV) and bond distance."""
    text = f"$J_\\mathrm{{{shell}}}$ = {j:.3f} meV"
    if dist is not None:
        text += f"\n$d$ = {dist:.2f} $\\AA$"
    ax.annotate(text, xy, color=color, ha="center", va="center", fontsize=8,
                zorder=5, bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                    ec=color, lw=1.0, alpha=0.9))


def write_exchange_doc(doc: dict, outdir: Path) -> None:
    """Write the ExchangeDocument to a JSON file."""
    out = outdir / "exchange_doc.json"
    with open(out, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_id", help="db_id of any job in the exchange flow")
    args = parser.parse_args()

    doc, flow_ids = load_exchange_doc(args.db_id)
    outdir = RESULTS_DIR / flow_ids[-1]
    outdir.mkdir(parents=True, exist_ok=True)

    plot_magnetization(doc, outdir, save_df=True)
    plot_exchange_graph(doc, outdir)
    write_exchange_doc(doc, outdir)


if __name__ == "__main__":
    main()
