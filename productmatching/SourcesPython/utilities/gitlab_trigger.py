import os
import requests
import tenacity
import logging

from collections import defaultdict
from time import sleep
from typing import List

TRIGGER_URL = "https://gitlab.heu.cz/api/v4/projects/657/trigger/pipeline"


def get_trigger_workflow_params(
    workflow_deploy_type: str,
    collector_categories: List[str],
    preceding_cache_address: List[str] = None,
    cache_address: List[str] = None,
    workflow_ids: List[str] = None,
):

    PARAMS = {
        "variables[WORKFLOW_DEPLOY_TYPE]": workflow_deploy_type,
        "variables[COLLECTOR_CATEGORIES]": "@@".join(collector_categories),
        "variables[PRECEDING_CACHE_ADDRESS]": "@@".join(preceding_cache_address),
        # by default leave empty cache to prevent using cache from value.yaml, rewritten if cache_address specified
        "variables[CACHE_ADDRES]": "",
    }

    if cache_address:
        PARAMS["variables[CACHE_ADDRESS]"] = "@@".join(cache_address)
    # optionally add workflow id, by defualt = 1 in gitlab predefined values
    if workflow_ids:
        PARAMS["variables[WORKFLOW_IDS]"] = "@@".join([str(wid) for wid in workflow_ids])

    return PARAMS


def get_trigger_matchapi_params(
    matchapi_ids: list = [],
    install: str = "false",
    uninstall: str = "false",
    target_envs: str = "stage"
):

    PARAMS = {
        "variables[MATCHAPI_DEPLOY_JOB]": target_envs,
        "variables[UNINSTALL]": str(uninstall),
        "variables[INSTALL]": str(install),
        "variables[IDS]": " ".join(matchapi_ids),
        }

    return PARAMS


@tenacity.retry(
    reraise=True,
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_random(min=0, max=2),
)
def list_pipeline_jobs(pipeline_id):
    LIST_JOBS_URL = f"https://gitlab.heu.cz/api/v4/projects/657/pipelines/{pipeline_id}/jobs"
    jobs = requests.post(url=LIST_JOBS_URL)
    return jobs.json()


@tenacity.retry(
    reraise=True,
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_random(min=0, max=2),
)
def retry_job(job_id):
    LIST_JOBS_URL = f"https://gitlab.heu.cz/api/v4/projects/657/jobs/{job_id}/retry"
    _ = requests.post(url=LIST_JOBS_URL)


def monitor_and_retry_jobs(pipeline_id: int) -> bool:
    """
    Checks the jobs of a pipeline and tries up to `max_failures` restarts for each failed job.
    Currently not used since we have the `retry` functionality in gitlab ci.
    """
    failures_counts = defaultdict(int)
    full_success, failure = False, False

    max_failures = 3
    while not (full_success or failure):
        jobs = list_pipeline_jobs(pipeline_id)
        full_success = True
        for job in jobs:
            if job["status"] == "failed":
                full_success = False
                jid = job["id"]
                failures_counts[jid] += 1
                if failures_counts[jid] > max_failures:
                    failure = True
                    break
                _ = retry_job(jid)
            if job["status"] != "success":
                full_success = False
        if full_success:
            break
        logging.info("Some of the pipeline jobs did not suceed yet, sleeping for 30s.")
        sleep(30)

    return full_success


@tenacity.retry(
    reraise=True,
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_random(min=0, max=2),
)
def get_pipeline_info(pipeline_id):
    PIPELINES_URL = f"https://gitlab.heu.cz/api/v4/projects/657/pipelines/{pipeline_id}"
    jobs = requests.get(url=PIPELINES_URL)
    return jobs.json()


def monitor_pipeline(pipeline_id):
    while True:
        pipeline_info = get_pipeline_info(pipeline_id)
        if pipeline_info["status"] in ["failed", "canceled"]:
            return False
        elif pipeline_info["status"] == "success":
            return True
        else:
            logging.info(f"Pipeline {pipeline_id} did not finish yet, sleeping for 30s")
            sleep(30)


@tenacity.retry(
    reraise=True,
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_random(min=0, max=2),
)
def trigger_pipeline(PARAMS: dict):

    logging.info(f"Received params: {PARAMS}")

    PARAMS["token"] = PARAMS.get("token", os.getenv("TRIGGER_TOKEN"))
    PARAMS["ref"] = PARAMS.get("ref", os.getenv("REF"))

    res = requests.post(url=TRIGGER_URL, params=PARAMS)

    return res
