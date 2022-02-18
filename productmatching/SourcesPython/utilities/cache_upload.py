import os
import json
import boto3
import logging
import argparse

from collections import defaultdict
from utilities.s3_utils import parse_s3_url, download_from_s3, upload_to_s3
from utilities.args import str_or_none

# ordered steps in main workflow, used for deleting all starting at specified step
# oncluding only the suffix with the relevant step info to be more robust against renaming of the steps
MAIN_WORKFLOW_STEPS_SUFFIX_ORDERED = [
    "collector_main",
    "collector_main_all",
    "retrain_data_download",
    "pfl_prevention",
    "embedding_dataset",
    "pmi_dataset",
    "fasttext_train",
    "candidates_retrieval",
    "dataset_split",
    "extract_attributes",
    "xgboostmatching_dataset",
    "xgboostmatching_train",
    "evaluation",
    "comparison",
]


def find_elements_by_suffix(steps: list, suffix: str, not_found_ok: bool = False):
    """
    Looks for elements in list ending with specified suffix.
    Returns found element and its index, asserts only one match found. Optionally ignore case when no element found
    """
    steps_ends_with = [st for st in steps if st.endswith(suffix)]
    if len(steps_ends_with) == 0 and not_found_ok:
        # do not raise Error
        logging.warning(f"No element ending on {suffix} not found between all possible steps")
        return None, None
    else:
        assert len(steps_ends_with) > 0, f"No element ending on {suffix} not found between all possible steps"
    assert len(steps_ends_with) == 1, f"Found {len(steps_ends_with)} element ending with {suffix}: {', '.join(steps_ends_with)}. Specify the element more precisely."
    found_step_ind = steps.index(steps_ends_with[0])

    return steps[found_step_ind], found_step_ind


# replace_artifacts_mapping = "collector_main_all-collector.tar.gz=s3://mlflow/459/e80bae1d6c2044358b07b714bfde4fbf/artifacts/matching_results.xlsx"
def edit_and_upload_cache(
    cache_url: str,
    new_cache_exp_id: str,
    new_cache_run_id: str,
    new_cache_name: str,
    delete_from_step: str = None,
    delete_only_steps: str = None,
    replace_artifacts_mapping: str = None,
):
    """
    Download cache from s3, delete (either invidual steps or specified step and all what follows) or replace artifacts url by new url.
    Upload the updated cache to specified location on s3. If no changes made, do not upload.
    Parameters:
    - cache_url: s3 url to cache to change
    - new_cache_exp_id, run_id, name: specification of new cache, used when pasting the new s3 url
    - delete-from-step: specify workflow step (given by its unique suffix, e.g. xgboostmatching_dataset) which will be deleted and all what follows (order specified in script); or None
    - delete-only-steps: specify individual steps to delete, seperated by @@ ; or None
    - replace-artifacts: specify artifacts where url should be replaced.
        Assumes structure 'step1-art_name1=new_url1@@step2-art_name2=new_url2@@...'
        'step{i}-' is needed only for artifacts present in multiple steps (e.g. collector.tar.gz), otherwise it is optional
    """
    # flag whether any chane were made in cache, upload only if True
    cache_changed = False

    # temporary path where to save cache
    temp_path = "temporary_cache.json"

    # download cache from s3
    temp_path = download_from_s3(url=cache_url, destination_path=temp_path)

    with open(temp_path, "r") as f:
        cache = json.load(f)

    # delete artifacts from all steps starting with specified step, order is given by MAIN_WORKFLOW_STEPS_SUFFIX_ORDERED
    if delete_from_step:
        cache_steps = list(cache.keys())
        # locate step from which to delete, look for steps ending with specified step name to overcome possible prefix renaming
        delete_step, delete_step_ind = find_elements_by_suffix(MAIN_WORKFLOW_STEPS_SUFFIX_ORDERED, delete_from_step)
        # delete
        for suffix_to_del in MAIN_WORKFLOW_STEPS_SUFFIX_ORDERED[delete_step_ind:]:
            # find the corresponding steps to delete, skip if there is no such step
            step_to_del, _ = find_elements_by_suffix(cache_steps, suffix_to_del, not_found_ok=True)
            if step_to_del:
                cache_changed = True
                cache.pop(step_to_del, None)
                logging.info(f"Deleted {step_to_del}")

    # delete only selected steps, do not care about the order of steps
    # assumes steps are separated by @@
    if delete_only_steps:
        to_del = delete_only_steps.split("@@")
        for td in to_del:
            # delete or log artifact not present in cache
            try:
                del cache[td]
                logging.info(f"Deleted {td}")
                cache_changed = True
            except KeyError:
                logging.warnin(f"Step {td} not present in cache")

    # replace artifacts if apecified
    # assumes structure 'step1-art_name1=new_url1@@step2-art_name2=new_url2@@...'
    # 'step{i}-' is needed only for artifacts present in multiple steps (e.g. collector.tar.gz), otherwise it is optional
    if replace_artifacts_mapping:
        # list all present artifacts and their step for easier manipulation
        art_name_to_step = defaultdict(list)
        for step, artifs in cache.items():
            for art in artifs.keys():
                art_name_to_step[art].append(step)

        to_replace = replace_artifacts_mapping.split("@@")
        for repl in to_replace:
            art_name, new_url = repl.split("=")
            if "-" in art_name:
                # step specified
                step, art_name = art_name.split("-")
                # replace
                cache[step][art_name] = new_url
                cache_changed = True
            else:
                # find it according to artifact name
                assert art_name in art_name_to_step.keys(), f"{art_name} not found in cache"
                step = art_name_to_step[art_name]
                assert len(step) == 1, f"Artifact {art_name} located in multiple steps ({', '.join(step)}), specify step where to replace"
                # replace
                cache[step[0]][art_name] = new_url
                cache_changed = True
            logging.info(f"Replaced url at {step} - {art_name} with {new_url}")

    if not cache_changed:
        logging.warning("Cache was not changed, not uploading")
        return

    # save cache
    with open(temp_path, "w") as f:
        json.dump(cache, f, indent=4, sort_keys=True)

    # get new url
    bucket, _, _ = parse_s3_url(cache_url)
    if not new_cache_name.endswith(".json"):
        new_cache_name += ".json"
    new_url = f"s3://{bucket}/{new_cache_exp_id}/{new_cache_run_id}/artifacts/{new_cache_name}"

    # upload cache
    upload_to_s3(new_url, temp_path)
    logging.info(f"Cache uploaded to {new_url}")

    return


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), force=True)
    parser = argparse.ArgumentParser()

    parser.add_argument("--cache-url", default=None, type=str)
    parser.add_argument("--new-cache-exp-id", required=True, type=str)
    parser.add_argument("--new-cache-run-id", required=True, type=str)
    parser.add_argument("--new-cache-name", required=True, type=str)
    parser.add_argument("--delete-from-step", required=False, type=str_or_none)
    parser.add_argument("--delete-only-steps", required=False, type=str_or_none)
    parser.add_argument("--replace-artifacts-mapping", required=False, type=str_or_none)

    args = parser.parse_args()

    logging.info(args)

    edit_and_upload_cache(
        cache_url=args.cache_url,
        new_cache_exp_id=args.new_cache_exp_id,
        new_cache_run_id=args.new_cache_run_id,
        new_cache_name=args.new_cache_name,
        delete_from_step=args.delete_from_step,
        delete_only_steps=args.delete_only_steps,
        replace_artifacts_mapping=args.replace_artifacts_mapping,
    )
