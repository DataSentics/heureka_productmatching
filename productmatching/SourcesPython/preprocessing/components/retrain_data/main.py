import os
import re
import argparse
import asyncio
import logging
import mlflow
import json
from datetime import datetime, timedelta
from itertools import chain
from multiprocessing.pool import Pool
from time import time
from uuid import uuid4

from utilities.component import compress
from utilities.galera import DbWorker
from utilities.loader import Corpus
from utilities.logger_to_file import log_to_file_and_terminal
from utilities.notify import notify
from utilities.remote_services import get_remote_services
from utilities.helpers import split_into_batches

from buttstrap.remote_services import RemoteServices
from matching_common.clients.cs2_client import CatalogueServiceClient


async def _process_decision_row(
    row: tuple, remote_services: RemoteServices, allowed_categories: set, offer_fields: list,
    product_fields: list, require_mismatch: bool = False, mct_checked: bool = False
):
    async with remote_services.get('catalogue').context as catalogue:
        offer_id = str(row[0])
        offers_data = []

        if require_mismatch:
            original_match = json.loads(row[1])["final_candidate"]

        try:
            # download offer
            async with remote_services.get('catalogue').context as catalogue:
                response = await CatalogueServiceClient(
                    catalogue_service=catalogue
                ).get_offers([offer_id], offer_fields, {"status": "all"})

            if not response:
                return
        except Exception as e:
            logging.warning(f"Exception while downloading data for offer {offer_id}: {e}")
            return

        offer_data = response[0]
        product_id = offer_data.get("product_id")
        if mct_checked:
            # in case of manual MCT control, we know the correct product match which might differ from the one in decisions
            # we pair an offer in wrong way -> stays undiscovered -> not in mismatched -> discovered in MCT -> gets here
            # TODO: might seem a little dirty with these indexes, but it is def more efficient than converting each row tuple to dict
            product_id = str(row[1])

        if not product_id:
            return

        if require_mismatch and original_match == product_id:
            return

        offers_data.append(offer_data)

        try:
            # candidate ~ product
            async with remote_services.get('catalogue').context as catalogue:
                response = await CatalogueServiceClient(
                    catalogue_service=catalogue
                ).get_products([product_id], product_fields, "all")

            product_data = response[0]
            product_category = product_data.get("category_id")
            if str(product_category) not in allowed_categories:
                # only certain model/categories are selected for retraining
                return

            if not product_data:
                # We need data for both product and offer
                return
        except Exception as e:
            logging.warning(f"Exception while downloading data for product {product_id}: {e}")
            return

        try:
            async with remote_services.get('catalogue').context as catalogue:
                response = await CatalogueServiceClient(
                    catalogue_service=catalogue
                ).get_products_offers(product_id)

            if not response:
                # sanity check, should not happen
                return
        except Exception as e:
            logging.warning(f"Exception while downloading offer_ids for product {product_id}: {e}")
            return

        try:
            product_offers = set(chain(*[offer["ids"].split(",") for offer in response])) - {offer_id}
            if product_offers:
                async with remote_services.get('catalogue').context as catalogue:
                    responses = await CatalogueServiceClient(
                        catalogue_service=catalogue
                    ).get_offers(list(product_offers), offer_fields, {"status": "all"})

                offers_data.extend([r for r in responses if r])

        except Exception as e:
            logging.warning(f"Exception while downloading offers data for product {product_id}: {e}")

        return (product_data, offers_data)


def _save_processed_rows(processed_rows: list, out_dir: str):
    uid = uuid4()
    file_products = os.path.join(out_dir, 'products', 'temp', f'products_{uid}.txt')
    path_offers = os.path.join(out_dir, 'offers')
    for pu in processed_rows:
        Corpus.save(file_products, json.dumps(pu[0]))
        product_id = pu[0]["id"]
        product_offers_path = os.path.join(path_offers, f"product.{product_id}.txt")
        for offer in pu[1]:
            Corpus.save(product_offers_path, json.dumps(offer))

    Corpus.close()


def _merge_product_files(out_dir):
    product_dir = os.path.join(out_dir, "products")
    temp_dir = os.path.join(product_dir, 'temp')
    path_products_duplicates = os.path.join(product_dir, "products_dup.txt")
    path_products = os.path.join(product_dir, "products.txt")
    product_files = os.listdir(temp_dir)
    with open(path_products_duplicates, "w") as fw:
        for pf in product_files:
            path = os.path.join(temp_dir, pf)
            with open(path, 'r') as fr:
                fw.write(fr.read())
            os.remove(path)
        os.rmdir(temp_dir)

    _dedup_product_file(path_products_duplicates, path_products)


def _dedup_product_file(path_in, path_out):
    product_ids = set()
    for line in Corpus.load(path_in):
        product_id = re.search(r"[0-9]+", line).group()
        if product_id in product_ids:
            pass
        else:
            product_ids.add(product_id)
            Corpus.save(path_out, line)
    os.remove(path_in)


async def process_model_decisions_table(
    args, table_name, fields, date_from, date_to, limit, out_dir, require_mismatch=False, mct_checked=False
):
    remote_services = await get_remote_services(['cs2', 'galera'])

    worker = DbWorker(remote_services)

    assert not (require_mismatch and mct_checked)
    st = time()

    if require_mismatch and "payload" not in fields:
        fields.append("payload")

    # TODO: change this according to the structure of the MCT table
    if mct_checked and "product_id" not in fields:
        fields.append("product_id")

    model_decisions = await worker.read_messages(table_name, date_from, date_to, fields, limit)
    logging.info(f"Read {len(model_decisions)} messages from table {table_name}.")
    # batch
    processed_decisions = []
    rows_in_batch = 10
    for batch_decisions in split_into_batches(model_decisions, rows_in_batch):
        processed_decisions_batch = await asyncio.gather(*[
            _process_decision_row(
                row, remote_services, args.categories, args.api_offer_fields, args.api_product_fields, require_mismatch, mct_checked
            )
            for row in batch_decisions
        ])
        processed_decisions.extend(processed_decisions_batch)

    processed_decisions = [pu for pu in processed_decisions if pu]
    if not processed_decisions:
        logging.info(f"No processed results from table {table_name}")
        return

    logging.info(f"saving {len(processed_decisions)} products from table {table_name}, took {round(time() - st, 2)}s")

    _save_processed_rows(processed_decisions, out_dir)
    await remote_services.close_all()


def run_process_model_decisions_table(kwargs):
    asyncio.run(process_model_decisions_table(**kwargs))


@notify
def main(args):
    DATE_FROM = datetime.today() - timedelta(args.retrain_gap)
    DATE_TO = datetime.today()
    LIMIT = None
    out_dir = os.path.join(args.data_directory, 'retrain_dataset')
    # TODO: big tables take longer time, 250K rows in unkown ~ 1K seconds
    # this might get out of control, and then there will arise a need to do further batching/multiprocessing on partitions of given table
    # the best option would be to have category column in galera tables, but that is not possible, since we need the category of currently paired product
    kwargs = [
        ({
            "args": args, "table_name": "matching_ng_unknown_match_cz", "fields": ["offer_id"],
            "date_from": DATE_FROM, "date_to": DATE_TO, "limit": LIMIT, "out_dir": out_dir,
        },),
        ({
            "args": args, "table_name": "matching_ng_item_new_candidate_cz", "fields": ["offer_id"],
            "date_from": DATE_FROM, "date_to": DATE_TO, "limit": LIMIT, "out_dir": out_dir,
        },),
        ({
            "args": args, "table_name": "matching_ng_item_matched_cz", "fields": ["offer_id", "payload"],
            "date_from": DATE_FROM, "date_to": DATE_TO, "limit": LIMIT, "out_dir": out_dir, "require_mismatch": True,
        },),
        # ({
        #     "args": args, "table_name": "XXX_MCT", "fields": ["offer_id", "product_id"],
        #     "date_from": DATE_FROM, "date_to": DATE_TO, "limit": LIMIT, "out_dir": out_dir, "mct_checked": True,
        # },),
    ]

    n_jobs = len(kwargs)
    with Pool(n_jobs) as pool:
        pool.starmap(run_process_model_decisions_table, kwargs)

    _merge_product_files(out_dir)

    Corpus.close()

    tar_file = os.path.join(args.data_directory, "retrain_dataset.tar.gz")
    compress(tar_file, out_dir)
    mlflow.log_artifact(str(tar_file))


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), force=True)
    parser = argparse.ArgumentParser()

    parser.add_argument("--categories", required=True)
    parser.add_argument("--data-directory", default="/data")
    parser.add_argument("--api-offer-fields", required=True)
    parser.add_argument("--api-product-fields", required=True)
    # TODO: start using this param or remove it completely
    parser.add_argument("--max-products-per-category", required=True)
    parser.add_argument("--retrain-gap", default=30)

    args = parser.parse_args()

    args.categories = set(args.categories.split(','))
    args.api_offer_fields = args.api_offer_fields.split(',')
    args.api_product_fields = args.api_product_fields.split(',')

    with mlflow.start_run():
        log_to_file_and_terminal(main, args)
