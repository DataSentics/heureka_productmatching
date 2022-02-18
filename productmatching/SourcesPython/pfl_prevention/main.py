import os
import argparse
import asyncio
import logging
import json
from collections import defaultdict

import mlflow

from utilities.component import process_inputs
from utilities.galera import DbWorker
from utilities.loader import Product, merge_collector_folders
from utilities.model_registry.client import MlflowRegistryClient
from utilities.notify import notify
from utilities.remote_services import get_remote_services

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), force=True)

MODEL_NAME = "matching_model"


def _get_ml_paired_offers_to_remove(args: argparse.Namespace, mct_evaluated_offers: set):
    """
    Removing offers paired by ML not checked by human from collector file/s.
    """
    offers_to_remove = defaultdict(list)
    n_offers = 0
    n_removed_offers = 0
    for product in Product.products(args.input_collector_products):
        for offer in Product.offers(args.input_collector_offers, product["id"]):
            n_offers += 1
            rt = offer.get("relation_type", {})
            if rt:
                if str(rt.get("value")) in args.ml_paired_offer_statuses and offer["id"] not in mct_evaluated_offers:
                    offers_to_remove[product["id"]].append(offer["id"])
                    n_removed_offers += 1

    logging.info(f"{n_removed_offers} offers were identified for removal")
    logging.info(f"{round(n_removed_offers / n_offers, 3) * 100} % of all offers were identified for removal")
    mlflow.log_metric("n_removed_offers", n_removed_offers)
    mlflow.log_metric("perc_removed_offers", round(n_removed_offers / n_offers, 2))

    return offers_to_remove


async def _get_galera_model_ids_from_categories(args, worker):
    """
    Finds all galera model ids for models serving specified categories.
    """
    mlflow_client = MlflowRegistryClient()
    tags = {"categories": ','.join(sorted(args.categories))}
    categories_models_info = mlflow_client.get_model_info_tags(MODEL_NAME, tags)

    galera_models_names = [f"{MODEL_NAME}_{mv.version}" for mv in categories_models_info]
    logging.info(f"Inspecting evaluated offers paired by following models: {galera_models_names}")
    mlflow.log_param("possible_galera_models_names", ', '.join(galera_models_names))

    js = "','"
    query = f"SELECT model_id FROM model WHERE model_name IN ('{js.join(galera_models_names)}')"
    galera_model_ids = await worker.execute_query(query)
    galera_model_ids = [str(tup[0]) for tup in galera_model_ids]
    mlflow.log_param("galera_model_ids", ', '.join(galera_model_ids))
    return galera_model_ids


async def _get_mct_evaluated_offers():
    """
    Finds all offers paired by ML and manually evaluated for models serving specified categories.
    """
    remote_services = await get_remote_services(['galera'])
    worker = DbWorker(remote_services)

    model_ids = await _get_galera_model_ids_from_categories(args, worker)

    if not model_ids:
        return

    select_evaluation = "SELECT matching_id FROM evaluation"
    select_matched = f"SELECT id, offer_id FROM matching_ng_item_matched WHERE model_id IN ({','.join(model_ids)})"
    query = f"SELECT offer_id FROM ({select_evaluation}) e JOIN ({select_matched}) mm ON e.matching_id = mm.id"
    mct_evaluated_offers = await worker.execute_query(query)
    mct_evaluated_offers = {str(tup[0]) for tup in mct_evaluated_offers}
    logging.info(f"Found {len(mct_evaluated_offers)} offers evaluated in MCT.")
    mlflow.log_metric("n_mct_evaluated_offers", len(mct_evaluated_offers))

    return mct_evaluated_offers


async def main(args):
    mct_evaluated_offers = await _get_mct_evaluated_offers()

    ml_paired_offers_to_remove = defaultdict(list)
    if mct_evaluated_offers:
        ml_paired_offers_to_remove = _get_ml_paired_offers_to_remove(args, mct_evaluated_offers)

    offers_to_remove_path = os.path.join(args.data_directory, "ml_paired_offers_to_remove.json")
    with open(offers_to_remove_path, "w") as fw:
        json.dump(ml_paired_offers_to_remove, fw)

    mlflow.log_artifact(offers_to_remove_path)


@notify
def positive_feedback_loop_prevention(args):
    asyncio.run(main(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--categories", required=True)
    parser.add_argument("--input-collector", required=True)
    parser.add_argument("--data-directory", default="/data")
    parser.add_argument("--ml-paired-offer-statuses", default="8")

    args = parser.parse_args()

    args.categories = args.categories.replace(' ', '').split(',')

    args.input_collector_files = merge_collector_folders(
        process_inputs(args.input_collector.split("@"), args.data_directory), args.data_directory
    )
    args.input_collector_products = args.input_collector_files + "/products"
    args.input_collector_offers = args.input_collector_files + "/offers"

    args.ml_paired_offer_statuses = args.ml_paired_offer_statuses.split(',')

    logging.info(args)

    with mlflow.start_run():
        positive_feedback_loop_prevention(args)
