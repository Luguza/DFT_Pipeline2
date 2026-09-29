from __future__ import annotations

from jobflow_remote import submit_flow
from mp_api.client import MPRester

from atomate2.common.flows.magnetism import MagneticOrderingsMaker
from atomate2.vasp.jobs.core import RelaxMaker, StaticMaker
from atomate2.vasp.sets.core import RelaxSetGenerator, StaticSetGenerator

MP_ID = ["mp-91"]  

if __name__ == "__main__":
    for mpid in MP_ID:
        with MPRester() as mpr:
            structure = mpr.get_structure_by_material_id(mpid)
        if "magmom" in structure.site_properties:
            structure.remove_site_property("magmom")

        static_maker = StaticMaker(
            input_set_generator=StaticSetGenerator(
                user_incar_settings={"EDIFF": 1e-6, "NELM": 300, "ALGO": "Normal"}
            )
        )
        relax_maker = RelaxMaker(
            input_set_generator=RelaxSetGenerator(
                user_incar_settings={"NELM": 300, "ALGO": "Normal"}
            )
        )

        flow = MagneticOrderingsMaker(
            name=mpid,
            static_maker=static_maker,
            relax_maker=relax_maker,
        ).make(structure)
        submit_flow(flow, worker="magnetism_justus2", project="dft_pipeline2")
