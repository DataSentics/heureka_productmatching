import os
import argparse
import asyncio
import json
import logging
import pandas as pd
from pathlib import Path
from collections import defaultdict
from time import time
from itertools import chain

from preprocessing.models import exact_match
from utilities.args import str_or_none, set_at_args
from utilities.attributes import Attributes
from utilities.cs2_downloader import CS2Downloader
from utilities.component import process_input, process_inputs, compress
from utilities.helpers import split_into_batches
from utilities.loader import Product, merge_collector_folders
from utilities.notify import notify
from utilities.remote_services import get_remote_services
from xgboostmatching.components.dataset.utils import create_examples, get_batch_indexes, XGB_ID_INFO_COLS, XGB_INFO_COLS
from xgboostmatching.utils import load_data_from_paths
from xgboostmatching.models.features import FEATURES_CONFIG

import mlflow


async def _get_products_data_proxy(downloader: CS2Downloader, batch_ids: list, batch_size: int):
    # might be somehow incorporated into CS2Downloader as a new method
    results = []
    async for products in downloader.products_download(batch_ids, batch_len=batch_size):
        results.extend(products)

    return results


async def _get_candidates_data(args, product_to_offers_to_candidates, batch_size: int = 15):
    """
    Get data for all candidates present in current batch.
    All of them should be present in the collector data when using only 'new' xgb input data.
    """
    candidates_ids_sources = chain(*[
        candidates for off_cands in product_to_offers_to_candidates.values() for candidates in off_cands.values()
    ])
    candidates_ids_to_source = {c[0]: c[1] for c in candidates_ids_sources}
    candidates_data = {}
    for product in Product.products(args.input_collector_products):
        if product["id"] in candidates_ids_to_source and candidates_ids_to_source[product["id"]] == "new":
            candidates_data[str(product["id"])] = product
    if args.preceding_input_collector_products:
        for product in Product.products(args.preceding_input_collector_products):
            if product["id"] in candidates_ids_to_source and candidates_ids_to_source[product["id"]] == "preceding":
                candidates_data[str(product["id"])] = product

    products_to_download = list(set(candidates_ids_to_source) - set(candidates_data))
    downloaded_products = []
    if products_to_download:
        remote_services = await get_remote_services(['cs2'])

        downloader = CS2Downloader(
            remote_services
        )

        batched_ids = split_into_batches(products_to_download, batch_size)
        coros = [_get_products_data_proxy(downloader, batch_ids, batch_size) for batch_ids in batched_ids]
        coros_batches = split_into_batches(coros, batch_size)
        for coro_batch in coros_batches:
            results = await asyncio.gather(*coro_batch)
            downloaded_products.extend(results[0])

    for product in downloaded_products:
        candidates_data[str(product["id"])] = product

    return candidates_data


def _get_extra_features_names(extra_features: dict):
    # get names of all features calculated in this run
    names = []
    for hl_feature, features_idxs in extra_features.items():
        raw_names_indexes = [FEATURES_CONFIG[hl_feature]["features"][i]["names_indexes"] for i in features_idxs]
        names += [tup[0] for rni in raw_names_indexes for tup in rni]

    return names


async def xgboostmatching_dataset_update(args: argparse.Namespace):
    job_spec = args.job_spec.split('/')
    job_no = int(job_spec[0])
    n_jobs = int(job_spec[1])

    # load data
    data = load_data_from_paths(args.input_datasets)
    data["source"] = "new"
    preceding_data = load_data_from_paths(args.preceding_input_datasets)
    if not preceding_data.empty:
        amess = "Column inconsistency between newly created xgb datasets and supplied preceding xgb datasets"
        assert set(data.columns) == set(preceding_data.columns), amess

        preceding_data["source"] = "preceding"

        data = pd.concat([data, preceding_data]).drop_duplicates(XGB_ID_INFO_COLS).reset_index(drop=True)

    product_offers_candidates_source = list({tup for tup in data[[*XGB_ID_INFO_COLS, "source"]].itertuples(index=False, name=None)})
    min_i, max_i = get_batch_indexes(job_no, len(product_offers_candidates_source), n_jobs)

    if min_i == max_i:
        extra_features_names = _get_extra_features_names(args.extra_features)
        # no products to be processed, output is an empty dataframe to prevent missing artifact
        logging.warning("No products after removal, resulting an empty dataframe")
        examples = pd.DataFrame(columns=XGB_INFO_COLS + extra_features_names)
    else:
        batch_product_offers_candidates = product_offers_candidates_source[min_i:max_i]
        product_to_offers_to_candidates_source = defaultdict(lambda: defaultdict(list))
        for qpid, oid, pid, src in batch_product_offers_candidates:
            product_to_offers_to_candidates_source[qpid][oid].append((pid, src))

        candidates_data = await _get_candidates_data(
            args,
            product_to_offers_to_candidates_source
        )

        namesimilarity = exact_match.ExactMatchModel(
            tok_norm_args=args.tok_norm_args
        )

        attributes = Attributes(args.input_attributes)

        # create final dataframe with features
        examples = await create_examples(
            args,
            namesimilarity,
            attributes,
            candidates_data,
            data[XGB_INFO_COLS],
            {},
            args.extra_features,
            False
        )

    directory = args.data_directory / f"xgboostmatching_dataset_extra_{job_no}"
    directory.mkdir(parents=True, exist_ok=True)

    examples.to_csv(str(directory / "xgboostmatching_dataset.csv"), index=False)

    tar_file = args.data_directory / f"xgboostmatching_dataset_extra_{job_no}.tar.gz"
    compress(tar_file, directory)

    mlflow.log_artifact(str(tar_file))


@notify
def main(args):
    asyncio.run(xgboostmatching_dataset_update(args))


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), force=True)
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-collector", required=True)
    parser.add_argument("--preceding-input-collector", type=str_or_none, default=None)
    parser.add_argument("--input-datasets", required=True)
    parser.add_argument("--preceding-input-datasets", type=str_or_none, default=None)
    parser.add_argument("--tok-norm-args", required=False)  # @@@ separated key=value pairs
    parser.add_argument("--input-attributes", required=True)
    parser.add_argument("--data-directory", type=Path, default="/data")
    parser.add_argument("--preceding-data-directory", default="/preceding_data")
    parser.add_argument("--job-spec", type=str, default="0/1")
    parser.add_argument("--extra-features", required=True)

    args = parser.parse_args()

    args.input_collector = Path(merge_collector_folders(
        process_inputs(args.input_collector.split("@"), args.data_directory), args.data_directory
    ))
    args.input_collector_products = args.input_collector / "products"
    args.input_collector_offers = args.input_collector / "offers"

    # expecting one file if any
    args.preceding_input_collector = process_input(args.preceding_input_collector, args.preceding_data_directory)
    args.preceding_input_collector_products = args.preceding_input_collector / "products" if args.preceding_input_collector else None
    args.preceding_input_collector_offers = args.preceding_input_collector / "offers" if args.preceding_input_collector else None

    args.input_datasets = process_inputs(args.input_datasets.split("@"), args.data_directory)
    # get preceding dataset uris or empty list if no specified
    args.preceding_input_datasets = args.preceding_input_datasets.split("@") if args.preceding_input_datasets else []
    # prepare data, different data directory required since it would replace the files from args.input_datasets
    args.preceding_input_datasets = process_inputs(args.preceding_input_datasets, args.preceding_data_directory)

    args.input_attributes = Path(process_input(args.input_attributes, args.data_directory))
    # mysteriously, a json string `xxx` is coverted into the form of '\'xxx\'' when passed through argparse
    args.extra_features = args.extra_features.strip("\'")
    args.extra_features = {
        fea: [int(i) for i in indexes] for fea, indexes in json.loads(args.extra_features).items()
    }

    args = set_at_args(args, 'tok_norm_args', args.data_directory)

    logging.info(args)

    with mlflow.start_run():
        main(args)
