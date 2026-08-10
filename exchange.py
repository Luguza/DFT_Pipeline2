"""
Run the exchange flow from a completed magnetic-orderings document.
This script is intended to be run on a cluster, after the magnetism.py flow has completed.

Options:

    -h --help: Show this help message and exit.

"""

from __future__ import annotations

import argparse
from jobflow_remote import submit_flow
from jobflow import run_locally
from jobflow_remote import JobController
from jobflow_remote.jobs.state import JobState

from atomate2.common.flows.exchange import ExchangeMaker
from monty.json import MontyDecoder

PROJECT = "dft_pipeline2"

def load_magnetic_orderings_doc(db_id: str) -> dict:
    """Load the raw MagneticOrderingsDocument dict from the flow containing job ``db_id``."""
    jc = JobController.from_project_name(PROJECT)
    flows = jc.get_flows_info(db_ids=[db_id], with_jobs_info=True)
    if not flows:
        raise SystemExit(f"No flow contains a job with db_id {db_id}.")
    flow = flows[0]
    try:
        idx = flow.job_names.index("postprocess orderings")
    except ValueError:
        raise SystemExit(
            f"Flow '{flow.name}' ({flow.flow_id}) has no 'postprocess orderings' job "
            "- is this a magnetic-orderings flow?"
        ) from None
    raw = jc.get_job_output(db_id=flow.db_ids[idx], load=True)
    if raw is None:
        raise SystemExit(
            f"Job '{flow.job_names[idx]}' (db_id {flow.db_ids[idx]}) has no output "
            "yet - has the flow finished?"
        )
    print(f"Using magnetic-orderings flow '{flow.name}' ({flow.flow_id}) with job db_id {flow.db_ids[idx]}.")
    return raw, flow.db_ids

def latest_ordering_doc():
    """Fetch the most recent completed MagneticOrderingsDocument from the jobstore."""
    jc = JobController.from_project_name(PROJECT)
    infos = jc.get_jobs_info(name="postprocess orderings", states=JobState.COMPLETED)
    if not infos:
        raise RuntimeError(
            "No completed magnetic-orderings flow found. "
            "Run magnetism.py and let it finish on the cluster first."
        )
    latest = max(infos, key=lambda i: i.updated_on)
    raw = jc.get_job_output(job_id=latest.uuid, job_index=latest.index, load=True)
    print(f"Using the most recent completed magnetic-orderings flow: {latest.name} ({latest.db_id})")
    return raw


def calc_exchange(ordering_doc):
    """Build the exchange flow from a completed magnetic-orderings document."""
    heisenberg_settings = {
        'cutoff': 5.0,
    }
    mc_settings = {
        'mc_box_size': 5.0,
        'equil_timesteps': 15000,
        'mc_timesteps': 30000,
        'save_inputs': True,
        'user_input_settings':{
            'start_t': 5,
            'end_t': 1000,
            'temp_increment': 5,
        }
    }
    return ExchangeMaker(
        heisenberg_settings=heisenberg_settings,
        run_vampire=True,
        mc_settings=mc_settings
    ).make_from_ordering_doc(ordering_doc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-id",
        "--db-id",
        type=str,
        help="The db_id of a job in a completed magnetic-orderings flow. "
        "If not provided, the most recent completed flow will be used.",
    )
    args = parser.parse_args()

    if args.db_id:
        raw, db_ids = load_magnetic_orderings_doc(args.db_id)
    else:
        raw = latest_ordering_doc()

    doc = MontyDecoder().process_decoded(raw)

    flow = calc_exchange(doc)
    submit_flow(flow, worker="exchange_justus2", project=PROJECT)
    # print("latest ordering doc:", doc)