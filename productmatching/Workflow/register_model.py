import os

from utilities import (
    write_stamp,
    wait_for_finish,
    run,
)


def register_model(cache, reinstall_matchapis_envs):
    write_stamp()

    collector_categories = os.environ.get("COLLECTOR_CATEGORIES")

    wait_for_finish(
        run(
            artifacts=[],
            entry_point="model_registration",
            parameters={
                "categories": collector_categories,
                "input_attributes": cache.get("preprocessing_extract_attributes", "attributes.json"),
                "tok_norm_args": "@@@".join([f'input_pmi={cache.get("preprocessing_pmi_dataset", "pmi.txt")}']),
                "input_xgb": cache.get("xgboostmatching_train", "best.xgb"),
                "data_directory": "/data",
                "thresholds_path": cache.get("evaluation", "thresholds.txt"),
                "cache_address": cache.cache_address,
                "reinstall_matchapi_envs": reinstall_matchapis_envs,
            },
            cache=cache,
            template_name="micro.yaml",
        )
    )
