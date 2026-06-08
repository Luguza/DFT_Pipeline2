from __future__ import annotations

from jobflow_remote import submit_flow
from jobflow_remote import JobController
from jobflow_remote.jobs.state import JobState

from atomate2.common.flows.exchange import ExchangeMaker
from monty.json import MontyDecoder

PROJECT = "dft_pipeline2"


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
    return MontyDecoder().process_decoded(raw)


def calc_exchange(ordering_doc):
    """Build the exchange flow from a completed magnetic-orderings document."""
    return ExchangeMaker(run_vampire=True).make_from_ordering_doc(ordering_doc)


if __name__ == "__main__":
    doc = latest_ordering_doc()
    flow = calc_exchange(doc)
    submit_flow(flow, worker="exchange_justus2", project=PROJECT)