from __future__ import annotations

from jobflow_remote import submit_flow
from pymatgen.core import Lattice, Structure

from atomate2.common.flows.magnetism import MagneticOrderingsMaker


if __name__ == "__main__":
    # rock-salt MnO as a minimal magnetic input structure
    structure = Structure(
        Lattice.cubic(4.445),
        ["Mn", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    flow = MagneticOrderingsMaker().make(structure)
    submit_flow(flow, worker="magnetism_justus2", project="dft_pipeline2")
