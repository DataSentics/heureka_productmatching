import argparse
import asyncio
import json
import logging
import os
from collections import defaultdict
from itertools import chain
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Iterable, List, Union

from matching_common.clients.provider import FAISS_CANDIDATES_SOURCE
import mlflow

from utilities.args import set_at_args
from utilities.candidates import get_candidate_provider
from utilities.candidates.candidates import CandidatesProvider
from utilities.component import process_input, process_inputs, compress
from utilities.cs2_downloader import CS2Downloader
from utilities.helpers import split_into_batches
from utilities.loader import Corpus, Product, merge_collector_folders
from utilities.notify import notify
from utilities.remote_services import get_remote_services


def clean_offers(args):
    for product_id, offer_ids in args.ml_paired_offers_to_remove.items():
        Product.delete_offers(args.input_collector_offers, args.input_collector_products, product_id, offer_ids)


def search_info_from_items(items_data: List[dict]):
    return [
        {'id': int(item["id"]), "match_name": item["match_name"]}
        for item in items_data if item
    ]


async def get_candidates(args, candidate_provider: CandidatesProvider, items_data: List[dict]):
    candidate_search_info = search_info_from_items(items_data)
    full_candidates = defaultdict(lambda: defaultdict(set))
    for source in args.candidates_sources:
        # {source: {offer_id: [product_id, ...]}}
        candidates = await candidate_provider.get_provider_candidates(
            source, candidate_search_info, limit=int(args.max_candidates),
            format_candidates=True, similarity_limit=args.similarity_limit,
            index_name="search-products-stage-cz", batch_size=20, value_field="match_name",
        )
        for id, candidates_ids in candidates.items():
            full_candidates[id][source] |= set(candidates_ids)

    return full_candidates


def get_offers_data_and_candidates(*args, **kwargs):
    return asyncio.run(_get_offers_data_and_candidates(*args, **kwargs))


async def _get_offers_data_and_candidates(args, products_batch: Iterable):
    remote_services = await get_remote_services(["cs2", "elastic"])
    # it is not necessary to download data for products indexed by faiss, we already have all the data we need
    candidate_provider = await get_candidate_provider(args, args.input_fasttext, None, remote_services)

    product_ids = set()
    all_candidates = set()
    for product in products_batch:
        product_offers_path = os.path.join(args.input_collector_offers, f"product.{product['id']}.txt")
        product_ids.add(str(product["id"]))
        offers = Product.offers(args.input_collector_offers, product["id"])
        candidates = await get_candidates(args, candidate_provider, offers)
        for offer in offers:
            candidates_sources = candidates.get(str(offer["id"]), {})
            candidates_ids = set(chain(*[candidates_sources.get(source, set()) for source in args.candidates_sources]))
            # slight formatting
            offer["candidates_sources"] = {k: list(v) for k, v in candidates_sources.items()}
            offer["candidates"] = list(candidates_ids)
            # rewrite current offers file
            Corpus.save(product_offers_path, json.dumps(offer))
            all_candidates |= candidates_ids

    Corpus.close()
    await remote_services.close_all()
    return product_ids, all_candidates


async def get_products_data(product_ids: List[Union[str, int]], fields: List[str], downloader: CS2Downloader):
    async for products in downloader.products_download(product_ids, return_params=False, fields=fields, batch_len=20):
        yield products


async def _get_products_data_batch(args, products_ids_batch: Iterable):
    remote_services = await get_remote_services(["cs2"])

    result = []
    if products_ids_batch:
        downloader = CS2Downloader(remote_services)
        async for product in get_products_data(products_ids_batch, args.api_product_fields, downloader):
            result.extend(product)

    await remote_services.close_all()

    return result


def get_products_data_batch(*args, **kwargs):
    return asyncio.run(_get_products_data_batch(*args, **kwargs))


async def main(args):
    # remove offers that might possibly cause positive model feedback
    clean_offers(args)
    # get offers data and candidates in multiple processes, loading fasttext and creating faiss index in each process
    n_processes = 4
    products_batches = split_into_batches(
        list(Product.products(args.input_collector_products)),
        n_batches=n_processes
    )
    args_list = [(args, batch) for batch in products_batches]

    with Pool(n_processes) as pool:
        results = pool.starmap(get_offers_data_and_candidates, args_list)

    product_ids = set()
    all_candidates = set()
    for r in results:
        product_ids |= r[0]
        all_candidates |= r[1]

    # in case multiprocessing turns out to cause any trouble use following code instead:
    # product_ids, all_candidates = await _get_offers_data_and_candidates(args, Product.products(args.input_collector_products))

    # download data for candidates which we don't have yet
    products_to_download = list(all_candidates - product_ids)
    if products_to_download:
        products_batches = split_into_batches(
            products_to_download,
            n_batches=n_processes
        )
        args_list = [(args, batch) for batch in products_batches]
        with Pool(n_processes) as pool:
            results = pool.starmap(get_products_data_batch, args_list)

        new_products_data = list(chain(*results))
        products_file_path = os.path.join(args.input_collector_products, "products.txt")
        for product in new_products_data:
            Corpus.save(products_file_path, json.dumps(product), 'a')

        Corpus.close()

    output_collector_dirname = "collector_candidates"
    # rename dir to keep the structure and be able to use it further in flow
    new_dir_name = os.path.join(
        os.path.dirname(args.input_collector),
        output_collector_dirname
    )

    os.rename(args.input_collector, new_dir_name)

    tar_file = os.path.join(args.data_directory, f"{output_collector_dirname}.tar.gz")
    compress(tar_file, new_dir_name)
    mlflow.log_artifact(str(tar_file))


@notify
def candidates_retrieval(args):
    import time
    st = time.time()
    asyncio.run(main(args))
    return f"{time.time() - st} s"


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), force=True)
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-collector", default=None)
    parser.add_argument("--api-product-fields", required=True)
    parser.add_argument("--input-fasttext", required=True)
    parser.add_argument("--tok-norm-args", required=False)  # @@@ separated key=value pairs
    parser.add_argument("--data-directory", default="/data")
    parser.add_argument("--similarity-limit", default=1)
    parser.add_argument("--max-candidates", default=10)
    parser.add_argument("--candidates-sources", default=FAISS_CANDIDATES_SOURCE)
    parser.add_argument("--ml-paired-offers-to-remove", default={})

    args = parser.parse_args()

    args.input_collector = Path(merge_collector_folders(
        process_inputs(args.input_collector.split("@"), args.data_directory), args.data_directory
    ))
    args.input_collector_products = args.input_collector / "products"
    args.input_collector_offers = args.input_collector / "offers"

    args.api_product_fields = args.api_product_fields.split(",")

    args.input_fasttext = process_input(args.input_fasttext, args.data_directory)

    args.data_directory = Path(args.data_directory)
    args.candidates_sources = args.candidates_sources.replace(' ', '').split(',')

    args = set_at_args(args, 'tok_norm_args', args.data_directory)

    if args.ml_paired_offers_to_remove:
        with open(process_input(args.ml_paired_offers_to_remove, args.data_directory), "r") as fr:
            args.ml_paired_offers_to_remove = json.load(fr)

    logging.info(args)

    with mlflow.start_run():
        candidates_retrieval(args)
