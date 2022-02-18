import os

from utilities import (
    write_stamp,
    wait_for_finish,
    run,
    Cache,
)


def mct_transfer():
    """
    Transforms decisions from matching excel info "candy-like" output messages and sends them to kafka
    for further use in MCT. Topic name is specified inside the script. 
    """
    write_stamp()

    cache_address = os.environ.get("CACHE_ADDRESS")
    cache = Cache(cache_address)

    collector_categories = os.environ.get("COLLECTOR_CATEGORIES")
    workflow_id = os.environ.get("WORKFLOW_ID")

    wait_for_finish(
        run(
            artifacts=[],
            entry_point="mct_transfer",
            parameters={
                "categories": collector_categories,
                "excel_path": cache.get("evaluation", "matching_results.xlsx"),
                "decisions_to_validate": "matched",
                "only_varying_matches": "true",
                "data_directory": "/data",
                "model_name": workflow_id,
                "cache_address": cache_address,
            },
            cache=cache,
            template_name="micro.yaml",
        )
    )
