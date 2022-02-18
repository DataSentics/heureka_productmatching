import os
import sys
import json
import calendar
import time
import datetime
import logging
import mlflow
import kubernetes
import mlflow.entities
import mlflow.projects.kubernetes

from typing import Optional, Union, List
from collections import defaultdict
from pathlib import Path

sys.path.append("/app/SourcesPython/utilities")
from component import process_input
from notify import format_mlflow_url, send_slack_message
from s3_utils import download_from_s3

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

MAIN_DIRECTORY = Path(os.path.dirname(os.path.realpath(__file__))).parent
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://catalogue-mlflow.stage.k8s.heu.cz")
CONTEXT = os.environ.get("CONTEXT", "production")
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", os.environ.get("IMAGE_TAG", "latest") + " - ProductMatching")

RETRAIN_FINISHED_MENTIONS = ["@petr.michal", "@jan.jeliga", "@ondrej.mekota"]

# thresholds to determine model registration and redeploy after retraining
# names correspond to metric names in evaluation step
metrics_redeploy_thr = {
    "overall precision_on_matched": 0.95,
    "new items precision_on_matched": 0.95,
    "overall coverage": 0.83,
    "new items coverage":  0.83
}


def check_retrained_model_metrics(cache, metrics_redeploy_thr):
    client = mlflow.tracking.MlflowClient()

    results_uri = cache.get("evaluation", "matching_results.xlsx")
    results_uri_spl = results_uri.split("/")
    # assumes uri structure e.g. 's3://mlflow/300/a01c4aaa0cce4a5986966480640220ee/artifacts/matching_results.xlsx'
    run_id = results_uri_spl[results_uri_spl.index("artifacts")-1]
    run = client.get_run(run_id)
    model_metrics = run.data.metrics

    # create slack message
    experiment_id = run.info.experiment_id
    mlflow_url = format_mlflow_url(experiment_id, run_id)
    msg = f"*Retraining model for {os.environ.get('COLLECTOR_CATEGORIES')} finished*. Evaluation sheet available at {mlflow_url} in artifact *matching_results.xlsx*.\n"

    run_register_model = True
    # check if the metrics are above specified thresholds
    missing_metrics = []
    for metr_name, metr_thr in metrics_redeploy_thr.items():
        if metr_name not in model_metrics.keys():
            run_register_model = False
            logging.warning(f"Metric {metr_name} not present in model metrics, skipping model registration and deploy")
            missing_metrics.append(metr_name)
        elif model_metrics[metr_name] < metr_thr:
            run_register_model = False
            logging.warning(f"Skipping registration and redeploy, metric {metr_name}={model_metrics[metr_name]} is below threshold {metr_thr}.")
            msg += f"Skipping registration and redeploy, metric *{metr_name}*={model_metrics[metr_name]} is *below {metr_thr}*.\n"

    if missing_metrics:
        msg += f"Skipping registration and redeploy, following metrics are missing: {','.join(missing_metrics)}.\n"
    elif run_register_model:
        msg += "Starting new model registration and matchapi reinstall.\n"

    # send slack message and notify users
    send_slack_message(
        msg=msg,
        user_mentions=RETRAIN_FINISHED_MENTIONS
    )

    return run_register_model


def replace_in_file(path: Union[str, Path], old: str, new: str):
    with open(path, "r") as f:
        filedata = f.read()

    filedata = filedata.replace(old, new)

    with open(path, "w") as f:
        f.write(filedata)


def wait_for_docker_container(wait: int = 60):
    DOCKER_HOST = os.environ.get("DOCKER_HOST")

    if DOCKER_HOST is not None:
        DOCKER_HOST = DOCKER_HOST.replace("unix:/", "")

        for _ in range(wait):
            if os.path.exists(DOCKER_HOST):
                break

            logging.info(f"{DOCKER_HOST} does not exist yet, waiting.")
            time.sleep(1)
    else:
        raise ValueError(f"{DOCKER_HOST} does not exist after {wait} seconds.")

    logging.info(f"{DOCKER_HOST} is ready.")


def get_config(repository_name: str, template_name: str = "low.yaml", branch: str = None):
    config = {
        "synchronous": False,
        "backend": "kubernetes",
        "backend_config": {
            "repository-uri": "registry.gitlab.heu.cz/catalogue/matching-ng/" + repository_name,
            "kube-job-template-path": str(MAIN_DIRECTORY / "Workflow" / "templates" / template_name)
        },
        # possible to use other branch than master (default with version=None)
        "version": branch,
    }

    if CONTEXT is not None:
        config["backend_config"]["kube-context"] = CONTEXT

    return config


def utc_now_timestamp() -> int:
    dt = datetime.datetime.utcnow()
    return calendar.timegm(dt.utctimetuple())


def write_stamp():
    with open('/tmp/stamp', 'w') as f:
        f.write(str(utc_now_timestamp()))


def get_status(submitted_job: mlflow.projects.kubernetes.KubernetesSubmittedRun):
    status = submitted_job._status
    kube_api = kubernetes.client.BatchV1Api()
    return status if mlflow.entities.RunStatus.is_terminated(status) else submitted_job._update_status(kube_api)


def wait_for_finish(*submitted_jobs: mlflow.projects.kubernetes.KubernetesSubmittedRun):
    for submitted_job, run_after_finish in submitted_jobs:
        if submitted_job is None:
            continue

        logging.info(f"Waiting for {submitted_job}.")

        while True:
            write_stamp()
            status = get_status(submitted_job)

            if status not in [mlflow.entities.RunStatus.FINISHED,
                              mlflow.entities.RunStatus.FAILED,
                              mlflow.entities.RunStatus.KILLED]:
                print(".", end=" ", flush=True)
                time.sleep(5)
                continue

            if status != mlflow.entities.RunStatus.FINISHED:
                logging.fatal("Job failed with status %s", mlflow.entities.RunStatus.to_string(status))
                sys.exit(1)

            break

        run_after_finish()
        logging.info(f"\nFinished {submitted_job}.")


def gitlab_uri(path):
    return "https://{username}:{password}@gitlab.heu.cz/{path}".format(
        username=os.environ["GITLAB_USERNAME"],
        password=os.environ["GITLAB_PASSWORD"],
        path=path
    )


def run(
    artifacts: list,
    entry_point: str,
    parameters: dict,
    cache: dict,
    uri: str = str(MAIN_DIRECTORY),
    repository_name: str = "productmatching",
    cache_suffix: str = "",
    template_name: str = "low.yaml",
    branch: str = None,
):
    if not isinstance(artifacts, list) and isinstance(artifacts, str):
        artifacts = [artifacts]

    write_stamp()
    logging.info(f"Starting {repository_name}, {entry_point}.")

    if len(artifacts) > 0 and all(cache.get(entry_point, a, repository_name=repository_name, suffix=cache_suffix) is not None for a in artifacts):
        logging.info(f"Skipping {repository_name}, {entry_point}.")
        return None, None

    submitted = mlflow.projects.run(
        uri=uri,
        entry_point=entry_point,
        parameters=parameters,
        **get_config(repository_name=repository_name, template_name=template_name, branch=branch),
    )

    active_run = mlflow.get_run(submitted.run_id)

    def run_after_finish():
        for artifact in artifacts:
            cache.set(entry_point, artifact, active_run.info.artifact_uri + "/" + artifact, repository_name=repository_name, suffix=cache_suffix)
        if artifacts:
            # get cache uri and update cache address, it is set as tag during model registration
            art_uri = mlflow.get_artifact_uri()
            cache.cache_address = f"{art_uri}/cache.json"

    return submitted, run_after_finish


class Cache:
    def __init__(self, cache_address: Optional[str] = None):
        if cache_address:
            self.cache_address = cache_address
            with open(process_input(cache_address, MAIN_DIRECTORY), "r") as f:
                self.cached = defaultdict(dict, json.load(f))

        else:
            self.cache_address = None
            self.cached = defaultdict(dict)

    def get_cache_name(self, entry_point: str, repository_name: str = "productmatching", suffix: str = ""):
        if suffix != "":
            suffix = f"_{suffix}"

        return f"{repository_name}_{entry_point}{suffix}"

    def set(self, entry_point: str, artifact: str, value, repository_name: str = "productmatching", suffix: str = ""):
        self.cached[self.get_cache_name(entry_point, repository_name, suffix=suffix)][artifact] = value

        with open("/tmp/cache.json", "w") as f:
            json.dump(self.cached, f, indent=4, sort_keys=True)

        mlflow.log_artifact("/tmp/cache.json")

    def get(self, entry_point: str, artifact: str, repository_name: str = "productmatching", suffix: str = ""):
        cache_name = self.get_cache_name(entry_point, repository_name, suffix=suffix)
        return self.cached.get(cache_name, {}).get(artifact)

    def list(self, entry_point: str, repository_name: str = "productmatching", suffix: str = "", key_prefix: List[str] = []):
        """
        List all artifacts in cache for given entry point, optionally filter those starting with specified prefix.
        Returns a dict of form {repository_name_artifact_name: artifact_uri} or empty dict if no artifacts found.
        """
        cache_name = self.get_cache_name(entry_point, repository_name, suffix=suffix)
        artifacts = self.cached.get(cache_name, {})
        if key_prefix:
            artifacts = {k: v for k, v in artifacts.items() if k.startswith(tuple(key_prefix))}
        return artifacts

    def to_properties(self):
        string = ""

        for cache_name, artifacts in self.cached.items():
            string += cache_name + "=" + ",".join(artifacts.values()) + "\n"

        return string


def add_to_cache(output_cache: Cache, source_cache: Cache, output_entry_point: str, source_entry_point: str, output_artifact_name_prefix: str, source_contains: List[str]):
    """
    Adds artifacts from source_entry point in source_cache to output cache. Artifacts from source cache are selected using desired prefixes specified in 'source_containes' param.
    Selected artifacts are optionally renamed using 'output_artifact_name_prefix' param and added to output cache.
    If the artifact links (s3 uris) are already in cache with possibly different name, they are not added.
    If the artifacts after rename would replace existing art., the are numbered with next numbers,
    E.g.: output_entry_point=source_entry_point='model_dataset', source_contains='model_dataset', output_artifact_name_prefix='preceding_model_dataset'
    will get artifacts from 'model_dataset' in source cache rename them to 'preceding_model_dataset' and add (those not already present) to 'model_dataset' entry_point in output cache.
    If there is already 'preceding_model_dataset_0' in output cache, newly added artifact's names will start from 'preceding_model_dataset_1'.
    """

    present_artifacts = output_cache.list(output_entry_point, key_prefix=[output_artifact_name_prefix]).values()

    artifacts_to_add = source_cache.list(source_entry_point, key_prefix=source_contains)
    # select only artifacts already not present
    artifacts_to_add = {k: v for k, v in artifacts_to_add.items() if v not in set(present_artifacts)}
    new_artifacts_index = len(present_artifacts)

    for art_name, art_uri in artifacts_to_add.items():
        new_name = art_name.split(".")[0]
        suffix = ".".join(art_name.split(".")[1:])
        new_name = f"{output_artifact_name_prefix}_{new_artifacts_index}.{suffix}"
        new_artifacts_index += 1
        output_cache.set(output_entry_point, new_name, art_uri)

    return output_cache


def get_static_datasets_addresses(categories: str, version: str = "-1") -> Optional[str]:
    # -1 for latest varsion, using the same version for all categories, versions are something like releases, use integers
    # TODO: enable per-category version usage in a non-breaking-change way
    info_file_name = os.getenv("STATIC_DATASETS_INFO_FILE", "category_dataset_info.json")
    s3_bucket = os.getenv("STATIC_DATASETS_BUCKET", "s3://ml-static-datasets")
    s3_path = f"{s3_bucket}/{info_file_name}"
    local_path = download_from_s3(s3_path, os.getcwd())

    with open(local_path, "r") as fr:
        sd_info = json.load(fr)

    os.remove(local_path)

    sd_categories_s3_addreses = {}
    for category in [c.strip() for c in categories.split(",")]:
        cat_info = sd_info.get(category)
        if not cat_info:
            sd_s3_addr = None
        else:
            if int(version) < 0:
                used_version = max([int(k) for k in cat_info["versions"].keys()])
                logging.info(f"Using latest version of static dataset for category {category}, version v{used_version}")
                sd_s3_addr = f"{s3_bucket}/{category}/v{used_version}/static_dataset_{category}.tar.gz"
            elif str(version) not in cat_info["versions"]:
                new_version = max(int(i) for i in cat_info["versions"].keys()) + 1
                logging.info(f"Will create a new version of static dataset for category {category}, version v{new_version}")
                sd_s3_addr = None
            else:
                sd_s3_addr = f"{s3_bucket}/{category}/v{version}/static_dataset_{category}.tar.gz"

        sd_categories_s3_addreses[category] = sd_s3_addr

    return sd_categories_s3_addreses
