import os
import logging
import mlflow

from clustering import cluster_categories
from register_model import register_model
from mct_transfer import mct_transfer
from utilities import (
    MAIN_DIRECTORY,
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    wait_for_docker_container,
    write_stamp,
    wait_for_finish,
    gitlab_uri,
    Cache,
    replace_in_file,
    run,
    metrics_redeploy_thr,
    check_retrained_model_metrics,
    add_to_cache,
    get_static_datasets_addresses,
)


def main_workflow() -> Cache:
    write_stamp()

    workflow_id = os.environ.get("WORKFLOW_ID")
    cache_address = os.environ.get("CACHE_ADDRESS", "")
    preceding_cache_address = os.environ.get("PRECEDING_CACHE_ADDRESS", "")
    collector_categories = os.environ.get("COLLECTOR_CATEGORIES")
    collector_statuses = set(os.environ.get("PRODUCT_STATUSES", "11").split(','))
    collector_max_products = int(os.environ.get("COLLECTOR_MAX_PRODUCTS", -1))

    output_cache = Cache(cache_address)
    preceding_cache = Cache(os.environ.get("PRECEDING_CACHE_ADDRESS"))

    retrain_run = int(os.getenv("RETRAIN_RUN", "0"))
    # check, we need both retrain data and preceding cache for retraining or none of it for training
    ass_mess = "Either provide both retrain data and preceding cache or none of them"
    assert (retrain_run and preceding_cache.cache_address) or (not preceding_cache.cache_address and retrain_run == 0), ass_mess

    # general info about workflow
    logging.info(f'Workflow id: {workflow_id}')
    logging.info(f'Cache address: {cache_address}')
    logging.info(f'Preceding cache: {preceding_cache_address}')
    logging.info(f'Retrain run: {retrain_run}')
    logging.info(f'Collector categories: {collector_categories}')

    mlflow.log_param("workflow_id", workflow_id)
    mlflow.log_param("categories", collector_categories)
    mlflow.log_param("statuses", collector_statuses)
    mlflow.log_param("max products", collector_max_products)
    mlflow.log_param("retrain_run", retrain_run)
    mlflow.log_param("cache_address", cache_address)
    mlflow.log_param("preceding_cache", preceding_cache_address)

    # totally arbitrary - should shorten the execution time of model_dataset and xgboostmatching_dataset approx. `n_micro_jobs` times
    n_micro_jobs = 10

    # excel for comparison with excel generated in this run or from actual cache
    validation_excel_to_compare = os.environ.get("VALIDATION_EXCEL_TO_COMPARE_ADDRESS")

    # Download products and offers
    collector_parameters = {
        "category": collector_categories,
        "data_directory": "/data",
        "api_url": "http://catalogue-catalogue-service2.cz.k8s.heu.cz",
        "api_offer_fields": "id,product_id,name,offer_name,match_name,price,parsed_attributes,attributes,ean,shop_id,image_url,external_image_url,url,description,relation_type",
        "api_product_fields": "id,category_id,name,slug,status,attributes,eans,shops,prices,producers,image_url,category.slug,description",
        "max_products_per_category": collector_max_products,
        # sometimes, CS2 has problems with too many requests, limit requesters to e.g. 2; collector's default value is 10
        "requesters": 10
    }

    # using static dataset for training and evaluation
    use_static_dataset = bool(int(os.environ.get("USE_STATIC_DATASET", "1")))
    static_datsets_versions = os.environ.get("STATIC_DATASET_VERSION", "-1")
    static_datasets_to_create = ""
    if use_static_dataset:
        static_dataset_s3_addresses = get_static_datasets_addresses(collector_categories, static_datsets_versions)
        # set "static_datasets_to_create = [some category ids]" if you want to create new static dataset for selected categories anyway
        static_datasets_to_create = ",".join([k for k, v in static_dataset_s3_addresses.items() if v is None])
        logging.info(f"static_datasets_to_create: {static_datasets_to_create}")

    if not retrain_run:
        # download new data only when we don't want to use static dataset
        # or when we want to use it and create new version for at least one category
        # we presume, that categories are always supperted in a cluster bundle (or less)
        # downloading data for all the categories in order to gain more variablity in candidates
        if not use_static_dataset or static_datasets_to_create:
            # possibly download these anyway and then just remove the products and their offers from matched
            # because a lot might have change since the original download of the old data
            if collector_statuses == {'11'}:
                wait_for_finish(
                    run(
                        artifacts=["collector.tar.gz"],
                        repository_name="collector",
                        entry_point="main",
                        uri=gitlab_uri("catalogue/matching-ng/collector.git"),
                        parameters={
                            "status": "all",
                            **collector_parameters,
                        },
                        cache=output_cache,
                        cache_suffix="all",
                    ),
                    run(
                        artifacts=["collector.tar.gz"],
                        repository_name="collector",
                        entry_point="main",
                        uri=gitlab_uri("catalogue/matching-ng/collector.git"),
                        parameters={
                            "status": "active",
                            **collector_parameters,
                        },
                        cache=output_cache,
                    ),
                )
                collector_data_uri = output_cache.get(
                    "main", "collector.tar.gz", repository_name="collector"
                )
                collector_data_all_uri = output_cache.get(
                    "main", "collector.tar.gz", repository_name="collector", suffix="all"
                )
            else:
                wait_for_finish(
                    run(
                        artifacts=["collector.tar.gz"],
                        repository_name="collector",
                        entry_point="main",
                        uri=gitlab_uri("catalogue/matching-ng/collector.git"),
                        parameters={
                            "status": "all",
                            **collector_parameters,
                        },
                        cache=output_cache,
                        cache_suffix="all",
                    ),
                )
                collector_data_uri = output_cache.get(
                    "main", "collector.tar.gz", repository_name="collector", suffix="all"
                )
                collector_data_all_uri = output_cache.get(
                    "main", "collector.tar.gz", repository_name="collector", suffix="all"
                )
        else:
            # using static datasets, not creating any
            collector_data_uri = "@".join(static_dataset_s3_addresses.values())
            collector_data_all_uri = collector_data_uri

    else:
        # download additional data
        # offers processed by ML but not matched or incorrectly matched and manually checked products/offers
        wait_for_finish(
            run(
                artifacts=["retrain_dataset.tar.gz"],
                entry_point="retrain_data_download",
                parameters={
                    # this is a bit clumsy, but calling it 'category' just doesn't seem right
                    "categories": collector_categories,
                    "api_url": "http://catalogue-catalogue-service2.cz.k8s.heu.cz/v1/",
                    **{k: v for k, v in collector_parameters.items() if k not in ["category", "api_url"]}
                },
                cache=output_cache,
                template_name="low.yaml",
            )
        )
        collector_data_uri = output_cache.get(
            "retrain_data_download", "retrain_dataset.tar.gz"
        )
        collector_data_all_uri = output_cache.get(
            "retrain_data_download", "retrain_dataset.tar.gz"
        )
        logging.info(
            f'Will work with retrain data from {collector_data_uri} and preceding cache {os.environ.get("PRECEDING_CACHE_ADDRESS")}.'
        )

    if not use_static_dataset or static_datasets_to_create:
        # positive feedback loop prevention

        wait_for_finish(
            run(
                artifacts=["ml_paired_offers_to_remove.json"],
                entry_point="pfl_prevention",
                parameters={
                    "categories": collector_categories,
                    "input_collector": collector_data_uri,
                    "data_directory": "/data",
                    "ml_paired_offer_statuses": "8",
                },
                cache=output_cache,
                template_name="low.yaml",
            ),
        )

        # create dataset for fasttext

        wait_for_finish(
            run(
                artifacts=["embedding_dataset.tar.gz"],
                entry_point="preprocessing_embedding_dataset",
                parameters={
                    "input_collector": collector_data_uri,
                    "data_directory": "/data",
                    "preceding_corpus": preceding_cache.get(
                        "preprocessing_embedding_dataset", "embedding_dataset.tar.gz"
                    ),
                },
                cache=output_cache,
                template_name="low.yaml",
            ),
        )

        # Calculate PMI and create embedding dataset with applied PMI and extract attributes from products

        wait_for_finish(
            run(
                artifacts=["pmi.txt", "pmi_embedding_dataset.tar.gz"],
                entry_point="preprocessing_pmi_dataset",
                parameters={
                    "input_embedding_dataset": output_cache.get(
                        "preprocessing_embedding_dataset", "embedding_dataset.tar.gz"
                    ),
                    "data_directory": "/data",
                },
                cache=output_cache,
            )
        )

        # Train fasttext
        # Still necessary for static dataset creation and the evaluation step if you intend to use FAISS as a candidate source

        wait_for_finish(
            run(
                artifacts=["fasttext.bin"],
                entry_point="preprocessing_fasttext_train",
                parameters={
                    "input_embedding_dataset": output_cache.get(
                        "preprocessing_pmi_dataset", "pmi_embedding_dataset.tar.gz",
                    ),
                    "data_directory": "/data",
                    "model": "skipgram",
                    "dim": 100,
                    "lr": 0.12,
                    "ws": 5,
                    "epoch": 10,
                    "min_count": 1,
                    "minn": 2,
                    "maxn": 3,
                    "neg": 15,
                },
                cache=output_cache,
                template_name="low.yaml",
            )
        )

        # get candidates and their data, replace collector address

        wait_for_finish(
            run(
                artifacts=["collector_candidates.tar.gz"],
                entry_point="candidates_retrieval",
                parameters={
                    "input_collector": collector_data_uri,
                    "api_product_fields": collector_parameters["api_product_fields"],
                    "input_fasttext": output_cache.get("preprocessing_fasttext_train", "fasttext.bin"),
                    "tok_norm_args": "@@@".join([f'input_pmi={output_cache.get("preprocessing_pmi_dataset", "pmi.txt")}']),
                    "data_directory": "/data",
                    "similarity_limit": os.environ.get("CANDIDATES_RETRIEVAL_SEARCH_THRESHOLD", 1),
                    "max_candidates": os.environ.get("CANDIDATES_RETRIEVAL_SEARCH_MAX_CANDIDATES", 10),
                    "candidates_sources": os.environ.get("CANDIDATES_RETRIEVAL_SOURCES", "elastic"),
                    "ml_paired_offers_to_remove": output_cache.get("pfl_prevention", "ml_paired_offers_to_remove.json"),
                },
                cache=output_cache,
                template_name="memory.yaml",
            ),
        )

        collector_data_uri = output_cache.get("candidates_retrieval", "collector_candidates.tar.gz")

    if static_datasets_to_create:
        current_addresses = [v for v in static_dataset_s3_addresses.values() if v is not None]
        wait_for_finish(
            run(
                artifacts=[],
                entry_point="static_dataset",
                parameters={
                    "datasets_to_create": current_addresses,
                    "categories": collector_categories,
                    "input_collector": collector_data_uri,
                    "data_directory": "/data",
                    "datasets_info_file": os.environ.get("STATIC_DATASETS_INFO_FILE", "category_dataset_info.json"),
                    "datasets_s3_bucket": os.environ.get("STATIC_DATASETS_BUCKET", "s3://ml-static-datasets"),
                    "upload_to_s3": "true",
                },
                cache=output_cache,
                template_name="low.yaml",
            ),
        )

        # retrieval of the frashly baked SDs' S3 addres
        static_dataset_s3_addresses = get_static_datasets_addresses(static_datasets_to_create, "-1")
        new_addreses = [v for v in static_dataset_s3_addresses.values() if v is not None]

        collector_data_uri = "@".join(current_addresses + new_addreses)
        collector_data_all_uri = collector_data_uri
        logging.info(f"new collector_data_uri: {collector_data_uri}")
        logging.info(f"new collector_data_all_uri: {collector_data_all_uri}")

    # Split dataset
    # Only for active data since all data are used only during fasttext training (which is not used now) and for attributes extraction

    wait_for_finish(
        run(
            artifacts=["test_items.list", "test_items_data.tar.gz", "train_collector_data.tar.gz"],
            entry_point="dataset_split",
            parameters={
                "input_collector": collector_data_uri,
                "categories": collector_categories,
                "preceding_test_items_ids_file": preceding_cache.get(
                    "dataset_split", "test_items.list"
                ),
                "data_directory": "/data",
                "train_size": 0.9,
                "max_test_items_in_category": 750,
            },
            cache=output_cache,
            template_name="low.yaml",
        )
    )

    # extract attributes

    wait_for_finish(
        run(
            artifacts=["attributes.json"],
            entry_point="preprocessing_extract_attributes",
            parameters={
                "input_collector": output_cache.get("dataset_split", "train_collector_data.tar.gz"),
                "preceding_attributes": preceding_cache.get(
                    "preprocessing_extract_attributes", "attributes.json"
                ),
                "data_directory": "/data",
            },
            cache=output_cache,
        ),
    )

    # Add preceding dataset to cache
    # training: no prec.cache, skipped
    # first retraining: preceding cache, add all renamed model datasets from preceding cache to (new) output cache
    # second second retraining: preceding cache (which was output in last retraining), add add all renamed model datasets (model_dataset + preceding) as new preceding
    if retrain_run:
        output_cache = add_to_cache(
            output_cache=output_cache,
            source_cache=preceding_cache,
            output_entry_point="preprocessing_model_dataset",
            source_entry_point="preprocessing_model_dataset",
            output_artifact_name_prefix="preceding_model_dataset",
            source_contains=["preceding_model_dataset", "model_dataset"],
        )

    # Create dataset for xgboost

    wait_for_finish(
        *[
            run(
                artifacts=[f"xgboostmatching_dataset_{i}.tar.gz"],
                entry_point="xgboostmatching_dataset",
                parameters={
                    "categories": collector_categories,
                    "input_collector": output_cache.get("dataset_split", "train_collector_data.tar.gz"),
                    "input_attributes": output_cache.get(
                        "preprocessing_extract_attributes", "attributes.json"
                    ),
                    "tok_norm_args": "@@@".join([f'input_pmi={output_cache.get("preprocessing_pmi_dataset", "pmi.txt")}']),
                    "input_fasttext": output_cache.get("preprocessing_fasttext_train", "fasttext.bin"),
                    "max_products": -1,
                    "data_directory": "/data",
                    "job_spec": f"{i}/{n_micro_jobs}",
                    "candidates_sources": os.environ.get("XGB_DATA_CANDIDATES_SOURCES", "elastic"),
                    "coros_batch_size": 5,
                    "products_frac": 1.0,
                    "min_product_offers": 2,
                    "max_sample_offers_per_product": os.environ.get("MAX_XGB_SAMPLE_OFFERS_PER_PRODUCT", "10")
                },
                cache=output_cache,
                template_name="low.yaml",
            )
            for i in range(n_micro_jobs)
        ]
    )

    # Add preceding dataset to cache
    if retrain_run:
        output_cache = add_to_cache(
            output_cache=output_cache,
            source_cache=preceding_cache,
            output_entry_point="xgboostmatching_dataset",
            source_entry_point="xgboostmatching_dataset",
            output_artifact_name_prefix="preceding_xgboostmatching_dataset",
            source_contains=["preceding_xgboostmatching_dataset", "xgboostmatching_dataset"],
        )
        output_cache = add_to_cache(
            output_cache=output_cache,
            source_cache=preceding_cache,
            output_entry_point="dataset_split",
            source_entry_point="dataset_split",
            output_artifact_name_prefix="preceding_train_collector_data",
            source_contains=["train_collector_data", "preceding_train_collector_data"],
        )

    input_datasets_extra = "None"
    if extra_features := os.getenv("EXTRA_XGB_FEATURES", ""):
        wait_for_finish(
            *[
                run(
                    artifacts=[f"xgboostmatching_dataset_extra_{i}.tar.gz"],
                    entry_point="xgboostmatching_dataset_update",
                    parameters={
                        "input_collector": output_cache.get("dataset_split", "train_collector_data.tar.gz"),
                        "preceding_input_collector": output_cache.get("dataset_split", "preceding_train_collector_data_0.tar.gz"),
                        "input_datasets": "@".join(output_cache.list("xgboostmatching_dataset", key_prefix=["xgboostmatching_dataset"]).values()),
                        "preceding_input_datasets": "@".join(
                            output_cache.list("xgboostmatching_dataset", key_prefix=["preceding_xgboostmatching_dataset"]).values()
                        ) or None,
                        "input_attributes": output_cache.get(
                            "preprocessing_extract_attributes", "attributes.json"
                        ),
                        "tok_norm_args": "@@@".join([f'input_pmi={output_cache.get("preprocessing_pmi_dataset", "pmi.txt")}']),
                        "data_directory": "/data",
                        "job_spec": f"{i}/{n_micro_jobs}",
                        "extra_features": extra_features,
                    },
                    cache=output_cache,
                    template_name="low.yaml",
                )
                for i in range(n_micro_jobs)
            ]
        )
        input_datasets_extra = "@".join(output_cache.list("xgboostmatching_dataset_update", key_prefix=["xgboostmatching_dataset_extra"]).values())
        logging.info(f"input_datasets_extra: {input_datasets_extra}")

    # Train xgboost

    XGB_PARAMETERS = "booster=gbtree,objective=binary:logistic,eval_metric=aucpr,learning_rate=0.287,max_depth=8,subsample=0.766,max_delta_step=9,reg_lambda=2.44,reg_alpha=0.05"

    wait_for_finish(
        run(
            artifacts=["best.xgb"],
            entry_point="xgboostmatching_train",
            parameters={
                "input_datasets": "@".join(output_cache.list("xgboostmatching_dataset", key_prefix=["xgboostmatching_dataset"]).values()),
                "input_datasets_extra": input_datasets_extra,
                "parameters": XGB_PARAMETERS,
                # fit without grid search in default
                "randomized_search_iter": -1,
                "data_directory": "/data",
                "iterations": 500,
                "train_size": 0.8,
                "n_components": "mle",
                "preceding_input_datasets": "@".join(
                    output_cache.list("xgboostmatching_dataset", key_prefix=["preceding_xgboostmatching_dataset"]).values()
                ) or None,
            },
            cache=output_cache,
            template_name="low.yaml",
        )
    )

    # Create validation excel

    wait_for_finish(
        run(
            artifacts=["matching_results.xlsx", "thresholds.txt"],
            entry_point="evaluation",
            parameters={
                "categories": collector_categories,
                "input_collector": collector_data_all_uri,
                # preferably, download the whole category/ies fro training and use data from collector
                "input_attributes": output_cache.get("preprocessing_extract_attributes", "attributes.json"),
                "tok_norm_args": "@@@".join([f'input_pmi={output_cache.get("preprocessing_pmi_dataset", "pmi.txt")}']),
                # fasttext used only for finding candidates with faiss
                "input_fasttext": output_cache.get("preprocessing_fasttext_train", "fasttext.bin"),
                "input_xgb": output_cache.get("xgboostmatching_train", "best.xgb"),
                "data_directory": "/data",
                "similarity_limit": os.environ.get("EVALUATION_SEARCH_THRESHOLD", 1),
                "max_candidates": os.environ.get("EVALUATION_SEARCH_MAX_CANDIDATES", 10),
                "candidates_sources": os.environ.get("EVALUATION_CANDIDATES_SOURCES", "faiss"),
                # during finetuning, first half of items is used to tune the threshold and second half is used for validation with new threshold
                # without finetuning, all items are used for validation with default thresholds defined in xgboost model
                "finetune_thresholds": "true",
                # testing items, either automatically created file or list specified by user
                "test_items_ids_file": output_cache.get("dataset_split", "test_items.list"),
                "test_items_data_file": output_cache.get("dataset_split", "test_items_data.tar.gz"),
                "preceding_test_items_data_file": preceding_cache.get("dataset_split", "test_items.list"),
                "preceding_test_items_ids_file": preceding_cache.get("dataset_split", "test_items_data.tar.gz"),
                "prioritize_status": "true",
                "unit_conversions": "true",
                "price_reject_a": "1000.0",
                "price_reject_b": "400.0",
                "price_reject_c": "2.5",
                "matched_confidence_threshold": os.getenv("MATCHED_CONFIDENCE_THRESHOLD", 0.75),
                "precision_confidence_threshold": os.getenv("PRECISION_CONFIDENCE_THRESHOLD", 0.95),
                "per_category_results_to_compare": os.getenv("PER_CATEGORY_RESULTS_TO_COMPARE_ADDRESS"),
            },
            cache=output_cache,
            template_name="low.yaml",
        )
    )

    # Compare models based on validation excels, run only if given second excel

    if validation_excel_to_compare:
        wait_for_finish(run(
            # leave it empty to enable multiple comparisons without need to edit the cache manually
            # it only checks whether to skip the step and add it to cache at the end
            artifacts=[],
            entry_point="comparison",
            parameters={
                "excel_path": output_cache.get("evaluation", "matching_results.xlsx"),
                "validation_excel_path": validation_excel_to_compare,
                "data_directory": "/data",
            },
            cache=output_cache,
            template_name="micro.yaml",
        ))

    return output_cache


if __name__ == "__main__":
    replace_in_file(
        MAIN_DIRECTORY / "MLProject", "latest", os.environ.get("IMAGE_TAG", "latest")
    )
    replace_in_file(
        MAIN_DIRECTORY / "MLProject",
        "name: ",
        "name: " + os.environ.get("IMAGE_TAG", "latest") + "-",
    )

    for template in ["low.yaml", "high.yaml", "memory.yaml", "micro.yaml"]:
        replace_in_file(
            MAIN_DIRECTORY / "Workflow" / "templates" / template,
            "username",
            os.environ.get("GITLAB_USER_NAME", "username").lower().replace(" ", "."),
        )
        for var in [
            "GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_SERVICE_ACCOUNT", "VAULT_TOKEN", "DEXTER_DB_PASSWORD",
            "REF", "TRIGGER_TOKEN", "PRODUCT_STATUSES", "USE_GCP_ELASTIC", "KUBE_ELASTIC_ADDRESS", "USE_ATTRIBUTE_CHECK", "KAFKA_BOOTSTRAP_SERVERS"
        ]:
            replace_in_file(
                MAIN_DIRECTORY / "Workflow" / "templates" / template,
                "@" + var,
                os.environ.get(var).strip()
            )

    wait_for_docker_container()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    wid = os.environ.get('WORKFLOW_ID')

    workflow_type = os.environ.get("WORKFLOW_TO_RUN", "main_workflow")
    assert workflow_type in ["main_workflow", "clustering", "mct_transfer"], f"Unknown workflow type {workflow_type}"
    logging.info(f"Running {workflow_type}")

    if workflow_type == "main_workflow":
        with mlflow.start_run(run_name=f"WorkFlow-{wid}"):
            cache = main_workflow()

        # training mode, optionally register model and do not redeploy matchapi
        run_register_model = int(os.getenv("REGISTER_MODEL", "0"))
        reinstall_matchapis_envs = ""

        if int(os.getenv("RETRAIN_RUN", "0")):
            # check model metrics, register if they are above specified thresholds and send slack notification
            run_register_model = check_retrained_model_metrics(cache, metrics_redeploy_thr)
            # redeploy newly registered model on specified environments
            reinstall_matchapis_envs = os.getenv("REINSTALL_MATCHAPIS_ENVS", "")

        # register to production, depends on MLFLOW_S3_ENDPOINT_URL and MLFLOW_TRACKING_URI env vars
        if run_register_model:
            mlflow.set_experiment("MODEL_REPO_REGISTRATION")
            with mlflow.start_run(run_name="model_registration"):
                register_model(cache, reinstall_matchapis_envs)

    elif workflow_type == "clustering":
        with mlflow.start_run(run_name=f"Clustering-{wid}"):
            cluster_categories()

    elif workflow_type == "mct_transfer":
        with mlflow.start_run(run_name=f"MCT-transfer{wid}"):
            mct_transfer()
