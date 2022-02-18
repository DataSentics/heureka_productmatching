import os
import sys
import argparse
import logging
import mlflow
import mlflow.pyfunc

from utilities.model_registry.client import MlflowRegistryClient
from utilities.model_registry.model import RegistryMatchingModelConstructor
from utilities.matchapi_operations import enable_matchapis
from utilities.notify import notify
from utilities.args import str_or_none, set_at_args


@notify
def main(args):
    # should be run only in kube
    # this is due to required artifact paths format and mlflow credentials
    registry_client = MlflowRegistryClient()

    logging.info("Model logging")
    model_name = "matching_model"  # might be a good idea to change the name for experiments
    model_mock = RegistryMatchingModelConstructor()
    code_files = ["SourcesPython/preprocessing", "SourcesPython/utilities", "SourcesPython/xgboostmatching"]
    # artifact_paths values should be in the format "S3://..."
    artifact_paths = {
        "attributes": args.input_attributes,
        "pmi": args.input_pmi,
        "xgb": args.input_xgb,
    }
    if args.thresholds_path and args.thresholds_path != 'None':
        artifact_paths["thresholds"] = args.thresholds_path

    registry_client.log_model(
        artifact_path=model_name,
        code_path=code_files,
        python_model=model_mock,
        artifacts=artifact_paths
    )

    logging.info("Model registration")
    model_info = registry_client.register_model(model_name)

    tags = {
        "categories": ",".join(sorted(args.categories.replace(' ', '').split(','))),
        "cache_address": args.cache_address,
    }

    registry_client.set_model_version_tags(model_name, model_info.version, tags)

    logging.info("Starting model retrieval for functionality testing.")

    # to ensure usage of registered code
    # this is of no real significance since the registered code is the very same as the one in SourcesPython
    modstodel = [m for m in sys.modules if "boostmatch" in m or "utilities.model_registry" in m]
    for md in modstodel:
        print(f"deleting {md}")
        del sys.modules[md]

    model = registry_client.get_model_by_version(model_name, model_info.version)

    logging.info("Testing retrieved model prediction.")

    from xgboostmatching.models.features import Product, Offer
    product = Product.parse_obj({'name': "Bufo Bufo", "category_id": "1235"})
    offer = Offer.parse_obj({"name": "Ufo Bufo", "price": 66.6, "shop": "endless pain"})
    logging.info(model.predict({"product": product, "offer": offer}))

    logging.info("Archiving the model currently present in 'Production'. This logic is only temporary.")
    prod_info = registry_client.get_model_info_stage(model_name, 'Production', tags)
    if prod_info:
        registry_client.model_stage_transition(model_name, prod_info.version, 'Archived')
        logging.info(f"Moving version {prod_info.version} to 'Archived stage'")
    logging.info(f"Moving the new model (version {model_info.version}) to 'Production' stage. This logic is only temporary.")
    registry_client.model_stage_transition(model_name, model_info.version, 'Production')

    new_model_info = registry_client.get_model_version_info(model_name, model_info.version)
    # reinstall matchapis
    if args.reinstall_matchapi_envs:
        logging.info(f"Starting with enabling and installation matchapi for {new_model_info.tags['categories']} on {args.reinstall_matchapi_envs}")

        enable_matchapis(
            categories_tags=[new_model_info.tags["categories"]],
            matchapis_install_envs=args.reinstall_matchapi_envs
        )


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), force=True)
    parser = argparse.ArgumentParser()

    parser.add_argument("--categories", required=True)
    parser.add_argument("--input-attributes", required=True)
    parser.add_argument("--tok-norm-args", required=False)  # @@@ separated key=value pairs
    parser.add_argument("--input-xgb", required=True)
    parser.add_argument("--data-directory", default="/data")
    parser.add_argument("--thresholds-path", default=None)
    parser.add_argument("--cache-address", default=None)
    parser.add_argument("--reinstall-matchapi-envs", type=str, default="")

    args = parser.parse_args()
    args = set_at_args(args, 'tok_norm_args', args.data_directory)

    logging.info(args)
    with mlflow.start_run():
        main(args)
