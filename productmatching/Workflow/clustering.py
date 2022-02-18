import os

from utilities import (
    wait_for_finish,
    Cache,
    run,
)


def cluster_categories():
    output_cache = Cache(os.environ.get("CACHE_ADDRESS"))
    collector_categories = os.environ.get("COLLECTOR_CATEGORIES")
    sbert_model = os.environ.get("SBERT_MODEL_ADDRESS")

    # Download data for categories clustering
    wait_for_finish(run(
        artifacts=["clustering_data.tar.gz"],
        entry_point="clustering_download",
        parameters={
            "categories": collector_categories,
            "max_products_in_category": 5_000,
            "data_directory": "/data"
        },
        cache=output_cache,
        template_name="low.yaml"
    ))

    # Cluster categories

    wait_for_finish(
        run(
            artifacts=["clustering_results.tar.gz"],
            entry_point="clustering_cluster",
            parameters={
                "input_dataset": output_cache.get("clustering_download", "clustering_data.tar.gz"),
                "input_transformer": sbert_model,
                "use_only_namesimilarity": 0,
                "min_clusters": 3,
                "max_clusters": 20,
                "diff_pct_thr": 0.5,
                "n_characteristic_words": 30,
                "n_components": -1,
            },
            cache=output_cache,
            template_name="memory.yaml"
        )
    )
