from __future__ import annotations

from jobflow import run_locally
from jobflow_remote import JobController
from jobflow_remote.jobs.state import JobState

from atomate2.common.flows.exchange import ExchangeMaker

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
    return jc.get_job_output(job_id=latest.uuid, job_index=latest.index, load=True)


def calc_exchange(ordering_doc):
    """Build the exchange flow from a completed magnetic-orderings document."""
    return ExchangeMaker(run_vampire=False).make_from_ordering_doc(ordering_doc)


if __name__ == "__main__":
    flow = calc_exchange(latest_ordering_doc())
    run_locally(flow)
