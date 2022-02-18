import os
import logging
import json
import typing as t

from utilities.gitlab_trigger import get_trigger_matchapi_params, trigger_pipeline
from utilities.s3_utils import download_from_s3, upload_to_s3

def load_matchapi_config(config_s3_address: str):

    matchapi_conf_path = download_from_s3(
        config_s3_address,
        '/data'
    )
    with open(matchapi_conf_path, 'r') as f:
        matchapi_conf = json.load(f)

    return matchapi_conf, matchapi_conf_path


def enable_matchapis(categories_tags: t.List[str], matchapis_install_envs: str = "stage"):
    config_s3_address = os.getenv("MATCHAPI_CONFIG_FILE", "s3://matchapi-data-cz/matchapi_id_categories_mapping.json")

    matchapi_conf, matchapi_conf_path = load_matchapi_config(config_s3_address)

    disabled_ids = matchapi_conf.get("DISABLED")
    disabled_ids = disabled_ids.split(',') if disabled_ids else []

    ids_to_enable = [di for di in disabled_ids if matchapi_conf[di] in categories_tags]

    # it could theretically happen the ids are already enabled for some reason, do nothing it that case
    if ids_to_enable:
        # upload new config before matchapi installation
        matchapi_conf_new = matchapi_conf.copy()
        matchapi_conf_new["DISABLED"] = ",".join(sorted([di for di in disabled_ids if di not in ids_to_enable]))

        with open(matchapi_conf_path, 'w') as f:
            json.dump(matchapi_conf_new, f)

        logging.warning(f"Uploading new matchapi conf {matchapi_conf_new}")
        upload_to_s3(config_s3_address, matchapi_conf_path)

        # install matchapis
        try:
            matchapi_trigger_params = get_trigger_matchapi_params(
                matchapi_ids=ids_to_enable,
                install="true",
                uninstall="true",
                target_envs=matchapis_install_envs,
            )
            _ = trigger_pipeline(matchapi_trigger_params)
        except Exception as err:
            # cannot install matchapis for some reason, upload old matchapi config
            logging.error(f"Failed to install matchapis: {err}. Returning to old matchapi to id config")

            with open(matchapi_conf_path, 'w') as f:
                json.dump(matchapi_conf, f)
            logging.warning(f"Uploading matchapi conf {matchapi_conf}")

            upload_to_s3(config_s3_address, matchapi_conf_path)

