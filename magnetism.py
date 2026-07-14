from __future__ import annotations

from jobflow_remote import submit_flow
from mp_api.client import MPRester

from atomate2.common.flows.magnetism import MagneticOrderingsMaker


if __name__ == "__main__":
    # rock-salt MnO (Fm-3m) from Materials Project as the magnetic parent
    with MPRester() as mpr:
        structure = mpr.get_structure_by_material_id("mp-19306")
    flow = MagneticOrderingsMaker(name="Fe3O4").make(structure)
    submit_flow(flow, worker="magnetism_justus2", project="dft_pipeline2")
