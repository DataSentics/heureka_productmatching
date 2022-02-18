import os
import random
import argparse
import asyncio
import logging
import itertools
import datetime
import pandas as pd
from pathlib import Path
import typing as t
from collections import defaultdict

from preprocessing.models import exact_match
from utilities.attributes import Attributes
from utilities.candidates import get_candidate_provider, CandidatesProvider
from utilities.cs2_downloader import CS2Downloader
from utilities.component import process_input, process_inputs, compress
from utilities.faiss_search import determine_build_index
from utilities.helpers import split_into_batches
from utilities.notify import notify
from utilities.args import set_at_args
from utilities.preprocessing import Pipeline
from utilities.loader import Product, merge_collector_folders
from utilities.remote_services import get_remote_services
from xgboostmatching.components.dataset.utils import create_examples, get_batch_indexes, XGB_INFO_COLS
from xgboostmatching.models import features

from matching_common.clients.provider import FAISS_CANDIDATES_SOURCE
import mlflow


def get_products_list(args: argparse.Namespace) -> t.List[dict]:
    random.seed(10)

    job_spec = args.job_spec.split('/')
    job_no = int(job_spec[0])
    n_jobs = int(job_spec[1])

    # there could be product without offers serving only as candidates, do not use them when creating xgb dataset
    # removal comes before batch split in order to get more balanced batches
    full_product_list = list(Product.products(args.input_collector / "products"))
    product_list = [
        pr for pr in full_product_list
        if Product.n_offers(args.input_collector / "offers", pr["id"]) >= args.min_product_offers
    ]
    logging.info(
        f"Removed {len(full_product_list)-len(product_list)} products with less than {args.min_product_offers} offers from total of {len(full_product_list)} products"
    )

    n_products = len(product_list)

    min_index, max_index = get_batch_indexes(job_no, n_products, n_jobs, args.max_products)
    logging.info(f"Processing products from index {min_index} to {max_index}")
    job_product_list = product_list[min_index:max_index]

    # use random selected sample of products per category
    if args.products_frac < 1.0 and len(job_product_list) > 0:
        orig_n_products = len(job_product_list)
        # list of product ids per category
        cat_to_pr_ids = defaultdict(set)
        for pr in job_product_list:
            cat_to_pr_ids[pr["category_id"]] |= {pr["id"]}

        product_ids_flt = set()
        for pr_ids in cat_to_pr_ids.values():
            n_products_cat = len(pr_ids)
            # number of filtered products, at least 1 in category
            n_products_flt = max(1, int(n_products_cat*args.products_frac))
            # filtered product ids
            product_ids_flt |= set(random.sample(pr_ids, n_products_flt))

        job_product_list = [pr for pr in job_product_list if pr["id"] in product_ids_flt]

        logging.info(f"Chose {args.products_frac} of original products from each category, in total {len(job_product_list)} out of {orig_n_products}")

    return job_product_list


async def _get_offer_candidates(args: argparse.Namespace, offer: dict, candidate_provider: CandidatesProvider):
    """
    Enriches "candidates_sources" field of the input `offer`.
    New candidates are retrieved from source specified in args.candidates_sources not present in the "candidates_sources" field.
    """
    candidates_by_sources = offer.get("candidates_sources", {})
    missing_sources = [c for c in args.candidates_sources if c not in candidates_by_sources]
    if missing_sources:
        candidate_search_info = {
            'id': int(offer["id"]), "match_name": offer["match_name"]
        }

        for source in missing_sources:
            candidates = await candidate_provider.get_provider_candidates(
                source, [candidate_search_info], limit=15,
                format_candidates=True, similarity_limit=5,
                index_name="search-products-stage-cz",
            )
            candidates_by_sources[source] = list(candidates.values())[0]

        offer["candidates_sources"] = candidates_by_sources

    return offer


def _select_product_offers(args: argparse.Namespace, offers: list):
    """
    Selects at most `args.max_sample_offers_per_product` offers with the highest number of candidates.
    """
    def _n_candidates(args, offer):
        return len(set(itertools.chain(*[offer.get("candidates_sources", {}).get(source, []) for source in args.candidates_sources])))
    if len(offers) <= args.max_sample_offers_per_product:
        return offers

    top_offers = sorted(offers, key=lambda o: _n_candidates(args, o), reverse=True)[:args.max_sample_offers_per_product]
    return top_offers


async def process_one_product(
    product: dict,
    full_product_dict: dict,
    args: argparse.Namespace,
    pipeline: Pipeline,
    downloader: CS2Downloader,
    candidate_provider: CandidatesProvider,
    cols: list,
):

    offers_path = args.input_collector / "offers"
    # save info about each tuple to dict, append them to dict and transform to dataframe at the end
    examples_dicts = []
    downloaded_product_data = defaultdict(dict)
    words = set(pipeline(product["name"]).split(" "))

    product_id = product["id"]
    offers = Product.offers(offers_path, product["id"])
    for offer in offers:
        offer = await _get_offer_candidates(args, offer, candidate_provider)

    n_offers_orig = len(offers)
    offers = _select_product_offers(args, offers)
    n_offers_new = len(offers)
    logging.info(f"product {product['id']}, n offers {n_offers_orig} -> {n_offers_new}")

    for offer in offers:
        examples_dicts.append(
            dict(zip(cols, [1.0, product_id, offer["id"], product_id, "product"]))
        )

        words |= set(pipeline(offer["match_name"]).split(" "))
        # get similar products from specified candidates clients, ommit the currently paired product
        similar_product_ids = set()

        # check for candidates in data, applies when using the static datset
        # possible to use all candidates when using "candidates" field containing all candidate ids
        candidates_by_sources = offer.get("candidates_sources", {})

        try:
            similar_product_ids = set(itertools.chain(*[candidates_by_sources[source] for source in args.candidates_sources])) - {str(product_id)}
        except KeyError as e:
            logging.exception(
                f"Offer {offer['id']} exception getting {args.candidates_sources} from candidates_by_sources {candidates_by_sources}"
            )
            raise e

        products_to_download = []
        for similar_product_id in list(similar_product_ids):
            sim_prod = full_product_dict.get(str(similar_product_id))
            if not sim_prod:
                products_to_download.append(similar_product_id)
                continue
            # use only if the product has enough offers or serves only as a candidate
            prod_n_offers = Product.n_offers(offers_path, similar_product_id)
            if prod_n_offers >= args.min_product_offers or prod_n_offers == 0:
                examples_dicts.append(
                    dict(zip(cols, [0.0, product_id, offer["id"], similar_product_id, "product"]))
                )
                words |= set(pipeline(sim_prod["name"]).split(" "))

        # download products data if any, only active products if not specified otherwise
        # passing empty list to `products_download` will download a batch of 'random' products with lowest ids
        if products_to_download:
            async for sp in downloader.products_download(
                    products_to_download,
                    status=[int(s) for s in os.environ.get("PRODUCT_STATUSES", "11").split(',')]
            ):
                if sp:
                    sim_prod = sp[0]
                    if sim_prod["offers_count"] >= args.min_product_offers:
                        examples_dicts.append(
                            dict(zip(cols, [0.0, product_id, offer["id"], sim_prod["id"], "product_downloaded"]))
                        )
                        downloaded_product_data[sim_prod["id"]] = sim_prod
                        words |= set(pipeline(sim_prod["name"]).split(" "))

    return examples_dicts, downloaded_product_data, words


async def create_and_download_tuples(
    args: argparse.Namespace,
    job_product_list: t.List[t.Dict],
    full_product_dict: t.Dict[str, dict],
    namesimilarity: exact_match.ExactMatchModel,
) -> t.Tuple[pd.DataFrame, t.DefaultDict]:

    cols = [*XGB_INFO_COLS, "product_type"]

    # save info about each tuple to dict, append them to dict and transform to dataframe at the end
    examples_dicts = []

    remote_services = await get_remote_services(['cs2', 'elastic'])

    downloader = CS2Downloader(
        remote_services
    )

    downloader_params = None
    build_index = False
    if FAISS_CANDIDATES_SOURCE in args.candidates_sources:
        build_index = determine_build_index(args.input_collector_offers)
    candidate_provider = await get_candidate_provider(
        args, args.input_fasttext, remote_services=remote_services, downloader_params=downloader_params, build_index=build_index
    )

    coros = []
    for product_index, product in enumerate(job_product_list):
        if args.max_products > -1 and product_index > args.max_products:
            break
        coros.append(
            process_one_product(
                product, full_product_dict, args, namesimilarity.pipeline, downloader, candidate_provider, cols
            )
        )

    # run coroutines in batches, running at once causes timeout
    logging.info(f"{len(coros)} coroutines")

    results = []
    for batch_coros in split_into_batches(coros, args.coros_batch_size):
        # run and await coroutines
        batch_results = await asyncio.gather(*batch_coros)
        results.extend(batch_results)
    logging.info("All batches processed.")

    # parse list of results
    examples_dicts = list(itertools.chain(*[one_or_res[0] for one_or_res in results]))
    # transform to dataframe
    examples = pd.DataFrame.from_dict(examples_dicts)
    downloaded_product_data = {k: v for one_pr_res in results for k, v in one_pr_res[1].items()}

    await candidate_provider.close()

    return examples, downloaded_product_data


async def xgboostmatching_dataset(args: argparse.Namespace):
    job_no = int(args.job_spec.split('/')[0])

    # get list of products for this job
    job_product_list = get_products_list(args)

    full_product_dict = {str(p["id"]): p for p in Product.products(args.input_collector / "products")}

    directory = args.data_directory / f"xgboostmatching_dataset_{job_no}"
    directory.mkdir(parents=True, exist_ok=True)

    if len(job_product_list) == 0:
        # no products to be processed, output is an empty dataframe to prevent missing artifact
        logging.warning("No products after removal, resulting an empty dataframe")
        examples = pd.DataFrame(columns=["label"] + features.all_features)
    else:
        namesimilarity = exact_match.ExactMatchModel(
            tok_norm_args=args.tok_norm_args
        )

        # create tuples with offer and product ids, download products not contained in collector data
        # get and normalize all unique words in titles
        examples_raw, downloaded_product_data = await create_and_download_tuples(
            args,
            job_product_list,
            full_product_dict,
            namesimilarity
        )

        attributes = Attributes(args.input_attributes)

        logging.info(f'Starting creating samples at {datetime.datetime.now().strftime("%H:%M")}')
        # create final dataframe with features
        examples = await create_examples(
            args,
            namesimilarity,
            attributes,
            full_product_dict,
            examples_raw,
            downloaded_product_data
        )

    examples.to_csv(str(directory / "xgboostmatching_dataset.csv"), index=False)

    tar_file = args.data_directory / f"xgboostmatching_dataset_{job_no}.tar.gz"
    compress(tar_file, directory)

    mlflow.log_artifact(str(tar_file))


@notify
def main(args):
    asyncio.run(xgboostmatching_dataset(args))


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), force=True)
    parser = argparse.ArgumentParser()

    parser.add_argument("--categories", required=True)
    parser.add_argument("--input-collector", required=True)
    parser.add_argument("--input-fasttext", required=True)
    parser.add_argument("--input-attributes", required=True)
    parser.add_argument("--tok-norm-args", required=False)  # @@@ separated key=value pairs
    parser.add_argument("--max-products", type=int, default=-1)
    parser.add_argument("--data-directory", type=Path, default="/data")
    parser.add_argument("--job-spec", type=str, default="0/1")
    parser.add_argument("--candidates-sources", type=str, default="elastic")
    parser.add_argument("--coros-batch-size", type=int, default=5)
    parser.add_argument("--products-frac", type=float, default=1.0)
    parser.add_argument("--min-product-offers", type=int, default=1)
    parser.add_argument("--max-sample-offers-per-product", type=int, default=10)
    args = parser.parse_args()

    args.categories = args.categories.replace(" ", "").split(",")
    args.input_collector = Path(merge_collector_folders(
        process_inputs(args.input_collector.split("@"), args.data_directory), args.data_directory
    ))
    args.input_collector_products = args.input_collector / "products"
    args.input_collector_offers = args.input_collector / "offers"
    args.candidates_sources = args.candidates_sources.replace(' ', '').split(',')

    if "faiss" in args.candidates_sources:
        args.input_fasttext = Path(process_input(args.input_fasttext, args.data_directory))

    args.input_attributes = Path(process_input(args.input_attributes, args.data_directory))

    args = set_at_args(args, 'tok_norm_args', args.data_directory)

    with mlflow.start_run():
        main(args)
