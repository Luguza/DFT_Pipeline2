"""Visualize the results of an exchange workflow.

Usage: python visualize_results.py [-id <db_id>]

``db_id`` is the jobflow-remote db_id of any job in the exchange flow (as shown
by ``jf job list``); if omitted, the most recently completed exchange flow is
used. Plots are written to ``results/<db_id>/``.

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
from jobflow_remote.jobs.state import FlowState
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
from monty.json import MontyDecoder
from pymatgen.analysis.magnetism.analyzer import CollinearMagneticStructureAnalyzer

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

# Magnetic ordering type -> colour. Fixed assignment, never cycled, so a given
# type keeps its colour across materials and across runs with different type mixes.
ORDERING_COLORS = {
    "FM": SERIES_COLORS[0],  # blue
    "AFM": SERIES_COLORS[7],  # orange
    "FiM": SERIES_COLORS[1],  # aqua
}


EXCHANGE_JOB = "build exchange doc"

def load_exchange_doc(db_id: str | None) -> dict:
    """Load the raw ExchangeDocument dict from the flow containing job ``db_id``.

    Without a ``db_id`` the most recently completed exchange flow is used, i.e.
    the newest COMPLETED flow that contains a ``build exchange doc`` job.
    """
    jc = JobController.from_project_name(PROJECT)
    if db_id is None:
        flows = jc.get_flows_info(
            states=FlowState.COMPLETED, sort=[("updated_on", -1)], with_jobs_info=True
        )
        flow = next((f for f in flows if EXCHANGE_JOB in f.job_names), None)
        if flow is None:
            raise SystemExit(
                f"No completed flow with a '{EXCHANGE_JOB}' job found in "
                f"project {PROJECT}."
            )
        print(f"Using latest completed exchange flow: {flow.name} ({flow.flow_id})")
    else:
        flows = jc.get_flows_info(db_ids=[db_id], with_jobs_info=True)
        if not flows:
            raise SystemExit(f"No flow contains a job with db_id {db_id}.")
        flow = flows[-1]
    try:
        idx = flow.job_names.index(EXCHANGE_JOB)
    except ValueError:
        raise SystemExit(
            f"Flow '{flow.name}' ({flow.flow_id}) has no '{EXCHANGE_JOB}' job "
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

    Like the caller, this reads the ground state ``magnetic_structures[0]``
    (magnetic ions only), which is what ``sublattice_ids`` is indexed against -
    ``structures`` retains the nonmagnetic ions and does not line up.

    Returns ``{material_id: (element, "up"|"down", M_sat_in_muB)}``.
    """
    hm = doc.get("heisenberg_model")
    if not hm:
        return {}
    structure = MontyDecoder().process_decoded(hm["magnetic_structures"][0])
    sublattice_ids = hm["sublattice_ids"][0]
    magmoms = structure.site_properties["magmom"]

    reps: dict[int, int] = {}  # material id (1-indexed) -> representative site
    mat_ids: dict[tuple[int, bool], int] = {}
    for site, (sub_id, magmom) in enumerate(zip(sublattice_ids, magmoms, strict=True)):
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


def plot_susceptibility(doc: dict, outdir: Path) -> None:
    """Plot the Vampire susceptibility curves; the peak of chi marks T_c.

    Mirrors ``plot_magnetization``: the mean susceptibility X_m is the bold
    curve and X_x/X_y/X_z its spatial components. The critical temperature is
    defined as the temperature of the X_m maximum, so the T_c line falls on the
    peak by construction.
    """
    if not doc.get("vampire_output"):
        print("Exchange doc has no vampire output; skipping susceptibility plot.")
        return

    vout = MontyDecoder().process_decoded(doc["vampire_output"])
    df = pd.read_json(StringIO(vout.parsed_out))
    tc = vout.critical_temp

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["T"], df["X_m"], color=SERIES_COLORS[0], lw=2, label="mean")
    for i, comp in enumerate("xyz"):
        ax.plot(
            df["T"],
            df[f"X_{comp}"],
            color=SERIES_COLORS[(i + 1) % len(SERIES_COLORS)],
            lw=2,
            alpha=0.5,
            label=rf"$\chi_{comp}$",
        )
    ax.axvline(tc, color=INK_MUTED, ls="--", lw=1.5, label=f"$T_c$ = {tc:.0f} K")

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"Susceptibility $\chi$")
    ax.set_title(f"{doc.get('formula_pretty', '')} Vampire Monte Carlo".strip())
    ax.grid(color=GRIDLINE, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()

    out = outdir / "susceptibility.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_exchange_graph(doc: dict, outdir: Path) -> None:
    """Draw the interaction graph: magnetic sites joined by J couplings.

    Nodes are the magnetic sites of the ground-state interaction graph
    (``HeisenbergModel.igraph``), coloured by spin direction and labelled with
    element, sublattice index and the site moment in muB.

    Edges carry the graph's own weights, which are what VAMPIRE integrates:
    ``J_ij |m_i m_j|`` in meV for the normalized-spin Hamiltonian, so the same
    fitted coupling gives different numbers on bonds between different moments.
    Each edge is annotated with that meV value and coloured by its coupling key
    ``{a}-{b}-{shell}`` (sublattice pair + neighbour shell); the legend maps the
    colour back to the moment-independent fit parameter in meV/muB^2 and the
    bond distance. Intra-site couplings are self-loops.
    """
    hm = doc.get("heisenberg_model")
    if not hm:
        print("Exchange doc has no heisenberg model; skipping exchange graph.")
        return

    ig = MontyDecoder().process_decoded(hm["igraph"])
    graph = ig.graph
    structure = ig.structure
    magmoms = structure.site_properties["magmom"]
    sublattice_ids = hm.get("sublattice_ids", [[]])[0]  # node index -> sublattice id

    # Map each fitted J value back to its coupling key "{a}-{b}-{shell}" and to
    # the (J, distance) shown in the legend. Colour is assigned per key.
    # ``dists`` is keyed by the same coupling key, so couplings sort by distance;
    # keys without one (the averaged "<J>") go last.
    dists = hm.get("dists", {})
    ex_params = {k: v for k, v in hm.get("ex_params", {}).items() if k != "E0"}
    keys = sorted(ex_params, key=lambda k: (dists.get(k, float("inf")), k))
    key_info = {k: (ex_params[k], dists.get(k)) for k in keys}
    key_colors = {
        k: SERIES_COLORS[(i + 1) % len(SERIES_COLORS)] for i, k in enumerate(keys)
    }

    def coupling_key(u: int, v: int, weight: float) -> str:
        """Identify which fitted parameter an edge weight came from.

        ``get_interaction_graph`` folds the site moments into the weight, so
        dividing them back out recovers the meV/muB^2 fit parameter to match
        against - comparing the weight itself never matches anything.
        """
        j = weight / abs(magmoms[u] * magmoms[v])
        key = min(keys, key=lambda k: abs(ex_params[k] - j), default="?")
        return key if key in ex_params and abs(ex_params[key] - j) < 1e-6 else "?"

    # Collapse the many periodic-image edges into one per (site pair, coupling).
    seen: set[tuple[int, int, str]] = set()
    edges: list[tuple[int, int, str, float]] = []
    for u, v, data in graph.edges(data=True):
        key = coupling_key(u, v, data["weight"])
        dedup = (min(u, v), max(u, v), key)
        if dedup in seen:
            continue
        seen.add(dedup)
        edges.append((u, v, key, data["weight"]))

    nodes = list(graph.nodes())
    pos = nx.circular_layout(nodes) if len(nodes) > 1 else {nodes[0]: np.zeros(2)}
    pos = {n: np.asarray(p, dtype=float) for n, p in pos.items()}
    centroid = np.mean(list(pos.values()), axis=0)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    def edge_label(xy: np.ndarray, weight: float, color: str) -> None:
        ax.annotate(f"{weight:+.1f}", xy, color=color, ha="center", va="center",
                    fontsize=7.5, fontweight="bold", zorder=5,
                    bbox={"boxstyle": "round,pad=0.15", "fc": "white",
                          "ec": "none", "alpha": 0.85})

    # Edges: every inter-site edge is bowed by the same base ``RAD`` so that
    # edges through the centre of the ring do not overlap their own labels;
    # parallel couplings between one pair fan out around it.
    RAD = 0.16
    inter = [e for e in edges if e[0] != e[1]]
    for node_pair in {(min(u, v), max(u, v)) for u, v, _, _ in inter}:
        group = [e for e in inter if (min(e[0], e[1]), max(e[0], e[1])) == node_pair]
        group.sort(key=lambda e: keys.index(e[2]) if e[2] in keys else 99)
        m = len(group)
        for k, (_u, _v, key, weight) in enumerate(group):
            # Draw low index -> high index so the bow direction is consistent.
            u, v = node_pair
            rad = RAD + 0.35 * (k - (m - 1) / 2)
            color = key_colors.get(key, INK_MUTED)
            ax.add_patch(FancyArrowPatch(
                pos[u], pos[v], connectionstyle=f"arc3,rad={rad}", arrowstyle="-",
                color=color, lw=2.5, shrinkA=30, shrinkB=30, zorder=2,
            ))
            # Midpoint of the arc3 quadratic Bezier: the chord midpoint pushed
            # perpendicular by half the control-point offset matplotlib uses.
            delta = pos[v] - pos[u]
            mid = (pos[u] + pos[v]) / 2 + 0.5 * rad * np.array([delta[1], -delta[0]])
            edge_label(mid, weight, color)

    for k, (u, _v, key, weight) in enumerate(e for e in edges if e[0] == e[1]):
        direction = pos[u] - centroid
        if np.linalg.norm(direction) < 1e-9:
            direction = np.array([np.cos(k), np.sin(k)])
        direction = direction / np.linalg.norm(direction)
        r = 0.22
        loop = pos[u] + direction * r
        color = key_colors.get(key, INK_MUTED)
        ax.add_patch(plt.Circle(loop, r, fill=False, zorder=2, lw=2.5, color=color))
        edge_label(loop + direction * r, weight, color)

    # Nodes: coloured by spin, labelled with element, sublattice index and the
    # site moment whose magnitude scales every edge weight touching the node.
    for n in nodes:
        up = magmoms[n] > 0
        ax.scatter(*pos[n], s=3200, zorder=3,
                   color=SERIES_COLORS[0] if up else SERIES_COLORS[5],
                   edgecolors="white", linewidths=2)
        sub = sublattice_ids[n] if n < len(sublattice_ids) else "?"
        ax.annotate(rf"$\mathrm{{{structure[n].specie.symbol}}}_{{{sub}}}$"
                    "\n"
                    rf"${magmoms[n]:+.2f}\,\mu_B$",
                    pos[n], color="white", ha="center", va="center",
                    fontsize=9, fontweight="bold", zorder=4, linespacing=1.4)

    # Legend: one coloured line per coupling, keyed by "{a}-{b}-{shell}", giving
    # the moment-independent fit parameter behind the meV numbers on the edges.
    handles = []
    for key in keys:
        j, dist = key_info[key]
        label = f"$J_\\mathrm{{{key}}}$ = {j:.3f} meV/$\\mu_B^2$"
        if dist is not None:
            label += f",  $d$ = {dist:.2f} $\\AA$"
        handles.append(Line2D([0], [0], color=key_colors[key], lw=2.5, label=label))
    if handles:
        ax.legend(handles=handles, frameon=False, loc="upper center",
                  bbox_to_anchor=(0.5, 0.0), ncol=1, fontsize=9,
                  handlelength=1.6, borderaxespad=0.0,
                  title=r"edge labels: $J_{ij}\,|m_i m_j|$ in meV (VAMPIRE ucf)",
                  title_fontsize=8)

    ax.set_title(f"{doc.get('formula_pretty', '')} exchange couplings".strip(), pad=2)
    ax.set_aspect("equal")
    ax.margins(0.18)
    ax.axis("off")
    fig.tight_layout()

    out = outdir / "exchange_graph.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_ordering_energies(doc: dict, outdir: Path) -> None:
    """Bar plot of the DFT energy of every ordering, coloured by ordering type.

    These are the energies the Heisenberg fit was built on, per magnetic ion and
    measured from the ground state, so the bars share a true zero baseline (the
    absolute values sit around -16.5 eV and would need a truncated axis).

    The type (FM / AFM / FiM) is classified by pymatgen's
    ``CollinearMagneticStructureAnalyzer`` from ``magnetic_structures``, which
    holds the magnetic ions only - induced moments on the nonmagnetic sites
    therefore cannot tip the classification. Bars are sorted by energy; the tick
    keeps the original index, which is what the ``ex_mat`` rows refer to.
    """
    hm = doc.get("heisenberg_model")
    if not hm:
        print("Exchange doc has no heisenberg model; skipping ordering energies.")
        return

    decoder = MontyDecoder()
    energies = hm["energies"]
    types = [
        CollinearMagneticStructureAnalyzer(
            decoder.process_decoded(s), make_primitive=False
        ).ordering.value
        for s in hm["magnetic_structures"]
    ]

    order = sorted(range(len(energies)), key=lambda i: energies[i])
    de = [(energies[i] - energies[order[0]]) * 1000 for i in order]  # meV per ion
    colors = [ORDERING_COLORS.get(types[i], INK_MUTED) for i in order]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(order))
    ax.bar(x, de, width=0.72, color=colors, zorder=2)

    # Type chip under every bar. The ground state sits at the zero baseline, so
    # its bar has no height to carry a colour; the chip row shows its type (and
    # every other one) independently of bar height.
    ax.scatter(
        x,
        [-0.028] * len(x),
        marker="s",
        s=42,
        color=colors,
        transform=ax.get_xaxis_transform(),
        clip_on=False,
        zorder=3,
    )
    ax.tick_params(axis="x", length=0, pad=14)

    # Value on every cap: identity is never colour-alone, and aqua sits below
    # 3:1 against the page, so the labels double as the required relief.
    for xi, d in zip(x, de, strict=True):
        ax.annotate(
            f"{d:.0f}",
            (xi, d),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            va="bottom",
            fontsize=8,
            color=INK_MUTED,
        )

    present = [t for t in ORDERING_COLORS if t in set(types)]
    handles = [Patch(facecolor=ORDERING_COLORS[t], label=t) for t in present]
    if any(t not in ORDERING_COLORS for t in types):
        handles.append(Patch(facecolor=INK_MUTED, label="other"))

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in order])
    ax.set_xlabel("Ordering index")
    ax.set_ylabel(r"$E - E_\mathrm{gs}$ (meV / magnetic ion)")
    ax.set_title(f"{doc.get('formula_pretty', '')} magnetic ordering energies".strip())
    ax.margins(y=0.12)  # headroom for the cap labels
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRIDLINE, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=handles, frameon=False)
    fig.tight_layout()

    out = outdir / "ordering_energies.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Wrote {out}")


def write_exchange_doc(doc: dict, outdir: Path) -> None:
    """Write the ExchangeDocument to a JSON file."""
    out = outdir / "exchange_doc.json"
    with open(out, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-id", 
        "--db_id", 
        default=None, 
        help="db_id of any job in the exchange flow "
             "(default: latest completed exchange flow)")
    args = parser.parse_args()

    doc, flow_ids = load_exchange_doc(args.db_id)
    outdir = RESULTS_DIR / min(flow_ids)
    outdir.mkdir(parents=True, exist_ok=True)

    plot_magnetization(doc, outdir, save_df=True)
    plot_susceptibility(doc, outdir)
    plot_exchange_graph(doc, outdir)
    plot_ordering_energies(doc, outdir)
    write_exchange_doc(doc, outdir)


if __name__ == "__main__":
    main()
