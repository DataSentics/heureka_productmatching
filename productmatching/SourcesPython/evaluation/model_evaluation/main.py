import argparse
import asyncio
import logging
import os
import numpy as np
import pandas as pd
from collections import defaultdict, namedtuple
from itertools import chain
from math import ceil
from multiprocessing.pool import Pool
from pathlib import Path
from pydantic import ValidationError
from scipy.stats import beta, norm
from sklearn.model_selection import train_test_split
from typing import List

from matching_common.clients.provider import FAISS_CANDIDATES_SOURCE
import mlflow

from evaluation.model_evaluation.utils import DocumentCreator, ThresholdsEvaluator
from preprocessing.models import exact_match
from utilities import loader
from utilities.attributes import (
    Attributes,
    parse_attributes,
    get_collector_product_offers_attributes,
    rinse_product_attributes
)
from utilities.args import str2bool, str_or_none, set_at_args
from utilities.candidates import CandidatesMonitor, get_candidate_provider
from utilities.component import process_input, process_inputs
from utilities.cs2_downloader import CS2Downloader
from utilities.faiss_search import determine_build_index
from utilities.loader import read_lines_with_int, merge_collector_folders
from utilities.notify import notify
from utilities.remote_services import get_remote_services
from xgboostmatching.models.model import XGBoostMatchingModel, Decision
from xgboostmatching.models.features import Product, Offer
from candy.logic.candy import Candy

Matches = namedtuple("Matches", "yes no unknown invalid invalid_offer")

PRODUCT_FIELDS = [
    "id", "category_id", "name", "prices", "slug", "category.slug",
    "attributes.id", "attributes.name", "attributes.value", "attributes.unit",
    "eans", "shops", "status", "producers"
]
PRODUCT_FIELDS_BASE = ["id", "category_id", "name", "slug", "category.slug", "status"]


def filter_item_ids(final_decisions_data: dict, paired_product_is_matched: list, item_ids_to_select: List[str] = None):
    final_decisions_data_sel = defaultdict(list)
    paired_product_is_matched_sel = []
    n_items_sel = 0
    for dec, result_dec in final_decisions_data.items():
        new_result_dec = [r for r in result_dec if str(r[0]["item_id"]) in item_ids_to_select]
        # add only if some offers are present, avoid getting key with empty list
        if new_result_dec:
            final_decisions_data_sel[dec] = new_result_dec
            n_items_sel += len(new_result_dec)

    paired_product_is_matched_sel = [pp for pp in paired_product_is_matched if str(pp[0]["item_id"]) in item_ids_to_select]

    return final_decisions_data_sel, paired_product_is_matched_sel, n_items_sel


def bayesian_bernoulli_confidence(sample_size: np.ndarray, n_successes: np.ndarray, thresholds: list = [], def_threshold: float = 0.95):
    """
    With flat prior, the posterior distribution of bernoulli distribution `p` parameter conditioned by sample of size `sample_size` with `n_successes`
    observed successes is Beta(a, b), where a = `n_successes + 1` and b = `sample_size - n_successes + 1`.
    We compute a value of the survival function (1 - cdf) at the `thresh` to estimate the probability that the true value of `p` is greater than `thresh`.
    We could probably come up with better prior than uniform, however, the tradeoff for simplicity of calculation is fair enough.
    This is equivalent to direct calculation of distribution of mean of independent bernoulli RVs.

    :param sample_sizes: array of sample sizes
    :param n_successes: array of number of successes
    :param thresh: list containing the values we want to compute the confidence of the real parameter being greater than
    """
    a_arr = n_successes + 1
    b_arr = sample_size - n_successes + 1

    if not thresholds:
        thresholds = [def_threshold] * len(sample_size)

    return [beta.sf(thresh, a=a, b=b) if thresh else None for a, b, thresh in zip(a_arr, b_arr, thresholds)]


def bernoulli_mle_confidence_intervals(sample_size: np.ndarray, n_successes: np.ndarray, alpha: float = .05):
    """
    Calculate MLE confidence intervals for parameter of bernoulli distributions.
    We use an estimation based on the logit transformation in order to keep the bounds of the CI within the [0, 1] interval.
    This interval should also have coverage closer to desired 95 % than classic normal approximation CI for the `p` parameter.

    :param sample_size: array of sample sizes
    :param n_successes: array of number of successes
    :param alpha: calculate the (1 - `alpha`) confidence intervals
    """
    def transform(x):
        return np.log(x / (1 - x))

    def inverse(x):
        return np.exp(x) / (1 + np.exp(x))

    phat = n_successes / sample_size
    information = sample_size * phat * (1 - phat)
    fihat = transform(phat)

    z_alpha = norm.ppf(1 - alpha / 2)
    lb_raw = fihat - z_alpha * np.sqrt(1 / information)
    ub_raw = fihat + z_alpha * np.sqrt(1 / information)
    lb, ub = inverse(lb_raw), inverse(ub_raw)

    return lb, ub


def _compare_previous_precision_matched_pct(dec_counts: pd.DataFrame, per_category_results_to_compare: str):
    if per_category_results_to_compare:
        pdf_pcrc = (
            dec_counts[["category"]]
            .astype({"category": str})
            .merge(
                pd.read_csv(per_category_results_to_compare)[["category", "matched_pct", "precision_on_matched"]],
                how="left",
                on="category",
            )
        )

        dec_counts["previous_matched_pct"] = pdf_pcrc["matched_pct"]
        dec_counts["matched_pct_confidence_over_previous"] = bayesian_bernoulli_confidence(
            dec_counts["n_items"], dec_counts["matched"], thresholds=dec_counts["previous_matched_pct"].tolist()
        )

        dec_counts["previous_precision_on_matched"] = pdf_pcrc["precision_on_matched"]
        dec_counts["precision_confidence_over_previous"] = bayesian_bernoulli_confidence(
            dec_counts["matched"], dec_counts["matched_as_orig"], thresholds=dec_counts["previous_precision_on_matched"].tolist()
        )

    return dec_counts


def _confidence_stats_to_per_category_results(
    dec_counts: pd.DataFrame,
    matched_thresh: float = .75,
    precision_thresh: float = .95,
    per_category_results_to_compare: str = None,

):
    m_str = str(int(matched_thresh * 100))
    dec_counts[f"matched_pct_confidence_over_{m_str}"] = bayesian_bernoulli_confidence(
        dec_counts["n_items"], dec_counts["matched"], def_threshold=matched_thresh
    )
    lb, ub = bernoulli_mle_confidence_intervals(dec_counts["n_items"], dec_counts["matched"])
    dec_counts["matched_pct_lower_confidence_bound"] = lb
    dec_counts["matched_pct_upper_confidence_bound"] = ub

    p_str = str(int(precision_thresh * 100))
    dec_counts[f"precision_confidence_over_{p_str}"] = bayesian_bernoulli_confidence(
        dec_counts["matched"], dec_counts["matched_as_orig"], def_threshold=matched_thresh
    )

    dec_counts = _compare_previous_precision_matched_pct(dec_counts, per_category_results_to_compare)

    lb, ub = bernoulli_mle_confidence_intervals(dec_counts["matched"], dec_counts["matched_as_orig"])
    dec_counts["precision_lower_confidence_bound"] = lb
    dec_counts["precision_upper_confidence_bound"] = ub

    dec_counts = dec_counts.fillna({
        "matched_pct_lower_confidence_bound": 0, "matched_pct_upper_confidence_bound": 1,
        "precision_lower_confidence_bound": 0, "precision_upper_confidence_bound": 1
    })

    return dec_counts


def calculate_business_metrics(
    final_decisions_data: dict,
    paired_product_is_matched: list,
    n_items: int,
    offers_data: dict = None,
    per_category: bool = False,
    item_ids_to_select: List[str] = None,
    matched_thresh: float = .75,
    precision_thresh: float = .95,
    per_category_results_to_compare: str = None,
):

    if item_ids_to_select:
        # filter results only for selected item ids and count them again
        final_decisions_data, paired_product_is_matched, n_items = filter_item_ids(
            final_decisions_data, paired_product_is_matched, item_ids_to_select
        )

    if not per_category:
        # only overall results, faster than calculating per category
        n_matched = len(final_decisions_data.get("matched", []))
        n_new_product = len(final_decisions_data.get("new_product", []))

        # coverage
        coverage = (n_matched + n_new_product) / n_items
        # percentage of matched
        matched_pct = n_matched / n_items
        # precision on matched
        if n_matched == 0:
            precision_on_matched = 0
        else:
            precision_on_matched = len(paired_product_is_matched) / n_matched

        return precision_on_matched, coverage, matched_pct
    else:
        # add matched_as_orig as new type of decision for easier manipulation
        all_decisions_data = final_decisions_data.copy()
        all_decisions_data["matched_as_orig"] = paired_product_is_matched

        # nested defaultdict, form {category1: {dec1: count, ...}, ...}
        cat_final_dec_counts = defaultdict(lambda: {
            "matched": 0,
            "new_product": 0,
            "unknown": 0,
            "matched_as_orig": 0
        })
        # count decisions per category
        for dec, results_dec in all_decisions_data.items():
            for item_res in results_dec:
                offer_id = item_res[0]["item_id"]
                offer_cat = offers_data[offer_id].get("product_category_id", "unknown")
                # add to counter
                cat_final_dec_counts[offer_cat][dec] += 1

        dec_counts = pd.DataFrame.from_dict(cat_final_dec_counts, orient="index").fillna(0)
        dec_counts["n_items"] = dec_counts[["matched", "new_product", "unknown"]].sum(axis=1)

        # add overall count and move index (category id) as first column for correct write to excel
        dec_counts = dec_counts.append(pd.Series(dec_counts.sum(axis=0), name="all")).reset_index()
        dec_counts.columns = ["category"] + list(dec_counts.columns[1:])

        dec_counts["coverage"] = (dec_counts["matched"] + dec_counts["new_product"]) / dec_counts["n_items"]
        dec_counts["matched_pct"] = dec_counts["matched"] / dec_counts["n_items"]
        dec_counts["precision_on_matched"] = (dec_counts["matched_as_orig"] / dec_counts["matched"]).fillna(0)

        dec_counts = _confidence_stats_to_per_category_results(
            dec_counts, matched_thresh, precision_thresh, per_category_results_to_compare
        )

        return dec_counts


def log_results(
    args,
    final_decisions_data: dict,
    paired_product_is_matched: list,
    n_items: int,
    offers_data: dict = None,
    per_category: bool = True,
    item_ids_to_select: List[str] = None,
    name_prefix: str = ""
):
    if name_prefix and not name_prefix.endswith(" "):
        # for better readability
        name_prefix = name_prefix + " "

    # business metrics, calculate them also per category
    per_category_results = calculate_business_metrics(
        final_decisions_data,
        paired_product_is_matched,
        n_items,
        offers_data,
        per_category,
        item_ids_to_select,
        args.matched_confidence_threshold,
        args.precision_confidence_threshold,
        args.per_category_results_to_compare,
    )

    per_category_res_path = str(Path(args.data_directory) / f'{name_prefix.replace(" ", "_").lower()}per_category_results.csv')
    per_category_results.to_csv(per_category_res_path, index=False)
    mlflow.log_artifact(per_category_res_path)

    # extract overall results
    m_str = str(int(args.matched_confidence_threshold * 100))
    p_str = str(int(args.precision_confidence_threshold * 100))

    overall_results = per_category_results[per_category_results["category"] == "all"]
    overall_coverage = float(overall_results["coverage"])
    overall_n_items = int(overall_results["n_items"])
    overall_matched_pct = float(overall_results["matched_pct"])
    overall_matched_pct_confidence_over_thr = float(overall_results[f"matched_pct_confidence_over_{m_str}"])
    overall_matched_pct_confidence_lb = float(overall_results["matched_pct_lower_confidence_bound"])
    overall_matched_pct_confidence_ub = float(overall_results["matched_pct_upper_confidence_bound"])
    overall_precision_on_matched = float(overall_results["precision_on_matched"])
    overall_precision_confidence_over_thr = float(overall_results[f"precision_confidence_over_{p_str}"])
    overall_precision_confidence_lb = float(overall_results["precision_lower_confidence_bound"])
    overall_precision_confidence_ub = float(overall_results["precision_upper_confidence_bound"])

    logging.info(f"{name_prefix}Number of items: {overall_n_items}")
    logging.info(f"{name_prefix}Coverage: {overall_coverage}")
    logging.info(f"{name_prefix}Percentage of matched: {overall_matched_pct}")
    logging.info(f"{name_prefix}Confidence of matched % being over {m_str}%: {overall_matched_pct_confidence_over_thr}")
    logging.info(f"{name_prefix}95% matched % MLE confidence interval: ({overall_matched_pct_confidence_lb}, {overall_matched_pct_confidence_ub})")
    logging.info(f"{name_prefix}Precision on matched: {overall_precision_on_matched}")
    logging.info(f"{name_prefix}Confidence of precision being over {p_str}%: {overall_precision_confidence_over_thr}")
    logging.info(f"{name_prefix}95% precision MLE confidence interval: ({overall_precision_confidence_lb}, {overall_precision_confidence_ub})")

    mlflow.log_metric(f"{name_prefix.lower()}n_items", n_items)
    mlflow.log_metric(f"{name_prefix.lower()}coverage", overall_coverage)
    mlflow.log_metric(f"{name_prefix.lower()}matched_pct", overall_matched_pct)
    mlflow.log_metric(f"{name_prefix.lower()}matched_pct_confidence_over_{m_str}", overall_matched_pct_confidence_over_thr)
    mlflow.log_metric(f"{name_prefix.lower()}matched_pct_95_confint_lower_bound", overall_matched_pct_confidence_lb)
    mlflow.log_metric(f"{name_prefix.lower()}matched_pct_95_confint_lower_bound", overall_matched_pct_confidence_ub)
    mlflow.log_metric(f"{name_prefix.lower()}precision_on_matched", overall_precision_on_matched)
    mlflow.log_metric(f"{name_prefix.lower()}precision_confidence_over_{p_str}", overall_precision_confidence_over_thr)
    mlflow.log_metric(f"{name_prefix.lower()}precision_95_confint_lower_bound", overall_precision_confidence_lb)
    mlflow.log_metric(f"{name_prefix.lower()}precision_95_confint_upper_bound", overall_precision_confidence_ub)

    return per_category_results


def rematch(new_threshold: float, orig_final_decisions, offers_data, n_items, prioritize_status: bool = False):
    """
    Recalculated existing decisions for new thresholds. Based on 'confidence' in Matches.
    Calculates and returns coverage together with precision on matched.
    """
    # tune and validation results, keep then in tuple for easier handling
    final_decisions_thr = defaultdict(list)
    paired_product_is_matched_thr = []
    invalid_offers = []

    for results_dec in orig_final_decisions.values():
        for item_res in results_dec:
            item_id = item_res[0]["item_id"]
            # store new decisions
            new_dec_per_item = defaultdict(list)

            # rematch based on new threshold
            for candidate_res in item_res:
                if candidate_res["confidence"]:
                    if candidate_res["confidence"] < new_threshold:
                        new_dec_per_item["no"].append(candidate_res)
                    else:
                        new_dec_per_item["yes"].append(candidate_res)
                else:
                    # do not change the decision
                    orig_decision = candidate_res['decision'].replace(' ', '_')
                    new_dec_per_item[orig_decision].append(candidate_res)

            if prioritize_status and len(new_dec_per_item["yes"]) > 0:
                new_dec_per_item["yes"], new_dec_per_item["no"], new_dec_per_item["unknown"] = Candy._prioritize_status(
                    new_dec_per_item["yes"], new_dec_per_item["no"], [], False
                )

            new_item_result = Matches(
                new_dec_per_item["yes"],
                new_dec_per_item["no"],
                [],  # unknown decision no longer supported in this case
                new_dec_per_item["invalid_data"],
                new_dec_per_item["invalid_offer_data"]
            )

            full_matches = new_item_result.yes + new_item_result.no + new_item_result.unknown + new_item_result.invalid
            invalid_offer = new_item_result.invalid_offer
            if invalid_offer:
                invalid_offers.append(invalid_offer)
            elif len(new_item_result.yes) == 1:
                final_decisions_thr['matched'].append(full_matches)
                if int(new_item_result.yes[0]["candidate_id"]) == int(offers_data[item_id]["product_id"]):
                    paired_product_is_matched_thr.append(full_matches)
            elif len(new_item_result.yes) == 0 and len(new_item_result.unknown) == 0:
                final_decisions_thr['new_product'].append(full_matches)
            elif len(new_item_result.yes) > 1 or (len(new_item_result.yes) == 0 and len(new_item_result.unknown) > 0):
                final_decisions_thr['unknown'].append(full_matches)

    logging.info(f"Rematching with threshold {new_threshold}:")
    logging.info([len(orig_final_decisions["matched"]), len(orig_final_decisions["new_product"]), len(orig_final_decisions["unknown"])])
    logging.info([len(final_decisions_thr["matched"]), len(final_decisions_thr["new_product"]), len(final_decisions_thr["unknown"])])

    return calculate_business_metrics(final_decisions_thr, paired_product_is_matched_thr, n_items, False)


def get_matching_model(args):
    namesimilarity = exact_match.ExactMatchModel(
        tok_norm_args=args.tok_norm_args
    )
    attributes = Attributes(
        from_=args.input_attributes,
    )
    xgboostmodel = XGBoostMatchingModel(
        xgboost_path=args.input_xgb,
        namesimilarity=namesimilarity,
        attributes=attributes,
        thresholds_path=args.thresholds_path if "thresholds_path" in args else None,
        unit_conversions=args.unit_conversions,
        price_reject_a=args.price_reject_a,
        price_reject_b=args.price_reject_b,
        price_reject_c=args.price_reject_c,
        tok_norm_args=args.tok_norm_args
    )

    return xgboostmodel


def _modify_offer_data(offer):
    """
    Modifies data of an offer in a way that it can be safely fed to `Offer` class and consequently to `XGBoostMatchingModel`.
    Any dict with "shop_id" field will pass without problem.
    """
    offer["shop"] = offer["shop_id"]
    offer["attributes"] = parse_attributes(offer, field="attributes")
    offer["parsed_attributes"] = parse_attributes(offer, field="parsed_attributes")
    return offer


async def get_offers_data(args, offer_ids_to_pair, downloader):
    offers_data = {}

    # get data on selected offers, download only for those not present in collector data or test items data
    for offers in loader.Product.offers_by_id(args.input_collector_offers, offer_ids_to_pair):
        for offer in offers:
            offers_data[str(offer["id"])] = _modify_offer_data(offer)

    # no need to check for presence in offer_ids_to_pair
    if args.preceding_test_items_data_file:
        for offer in loader.load_json(args.preceding_test_items_data_file):
            if str(offer["id"]) in offer_ids_to_pair:
                offers_data[str(offer["id"])] = _modify_offer_data(offer)

    if args.test_items_data_file:
        for offer in loader.load_json(args.test_items_data_file):
            if str(offer["id"]) in offer_ids_to_pair:
                offers_data[str(offer["id"])] = _modify_offer_data(offer)

    offer_ids_to_download = list(set(offer_ids_to_pair) - set(offers_data.keys()))

    if offer_ids_to_download:
        async for offers in downloader.offers_download(offer_ids_to_download):
            for offer in offers:
                # offer without product_id sometime occurs and causes KeyError later
                if offer and offer["product_id"]:
                    offers_data[str(offer["id"])] = _modify_offer_data(offer)

    # no need to get candidates for offers present in test_items_candidates_info
    candidate_search_info = []
    for offer in offers_data.values():
        if "candidates" not in offer:
            candidate_search_info.append(
                {
                    'id': int(offer["id"]),
                    "match_name": offer["match_name"],
                    "id_paired": str(offer["product_id"])
                }
            )

    return offers_data, candidate_search_info


async def get_products_data(args, offers_data: dict, offer_id_to_product_ids: dict, downloader: CS2Downloader):
    async def format_product(args, product, downloader, categories_info):
        product["attributes"] = parse_attributes(product)
        product["offers_attributes"] = get_collector_product_offers_attributes(args, product["id"], True)
        product["category_slug"] = product.get("category", {}).get("slug", "")
        product["status"] = product.get("status", {}).get("id", None)
        category = product["category_id"]
        if category not in categories_info:
            category_info = await downloader.category_info(category, ["ean_required", "unique_names", "long_tail"])
            categories_info[category] = category_info[0]
        product.update(categories_info[category])
        return product

    product_ids_to_dl = set(chain(*[ids for ids in offer_id_to_product_ids.values()]))

    products_data = {}
    categories_info = {}

    for product in loader.Product.products(args.input_collector_products):
        spid = str(product["id"])
        if spid in product_ids_to_dl:
            product_ids_to_dl.remove(spid)
            products_data[spid] = await format_product(args, product, downloader, categories_info)

    for offer_id, product_ids in offer_id_to_product_ids.items():
        dl_product_ids = list(set(product_ids) & product_ids_to_dl)
        if dl_product_ids:
            async for _product_param in downloader.products_download(dl_product_ids, return_params=True, fields=PRODUCT_FIELDS):
                _product, param = _product_param[0], _product_param[1]
                if _product:
                    product = await format_product(args, _product[0], downloader, categories_info)
                else:
                    product = {}
                products_data[str(param["id"][0])] = product

        original_id = offers_data[offer_id]["product_id"]
        async for _product in downloader.products_download([original_id], fields=PRODUCT_FIELDS_BASE):
            product = _product[0] if _product else {}
            for n in ["name", "category_id", "slug"]:
                offers_data[offer_id][f"product_{n}"] = product.get(n, None)
            # category slug is nested in category
            offers_data[offer_id]["product_category_slug"] = product.get("category", {}).get("slug", None)

    return products_data


def merge_unformatted_candidates(unformatted_candidates_by_source):
    merged_unformatted_candidates = defaultdict(dict)
    for source, unformatted_candidates in unformatted_candidates_by_source.items():
        for offer_id, candidates_dict in unformatted_candidates.items():
            s_offer_id = str(offer_id)
            if s_offer_id in merged_unformatted_candidates:
                for candidate_id, candidate in candidates_dict.items():
                    s_candidate_id = str(candidate_id)
                    if s_candidate_id in merged_unformatted_candidates[s_offer_id]:
                        merged_unformatted_candidates[s_offer_id][s_candidate_id].source.append(source)
                        dist = merged_unformatted_candidates[s_offer_id][s_candidate_id].distance
                        rele = merged_unformatted_candidates[s_offer_id][s_candidate_id].relevance
                        merged_unformatted_candidates[s_offer_id][s_candidate_id].distance = dist if dist else candidate.distance
                        merged_unformatted_candidates[s_offer_id][s_candidate_id].relevance = rele if rele else candidate.relevance
                    else:
                        merged_unformatted_candidates[s_offer_id][candidate_id] = candidate
            else:
                merged_unformatted_candidates[s_offer_id] = candidates_dict

    return merged_unformatted_candidates


async def get_candidates(args, offers_data, candidate_search_info, candidate_provider, batch_size):
    unformatted_candidates_by_source = {}
    candidates_by_source = {}

    # This takes eternity in kube during testing when using faiss
    for source in args.candidates_sources:
        unformatted_candidates = await candidate_provider.get_provider_candidates(
            source, candidate_search_info, limit=int(args.max_candidates),
            format_candidates=False, similarity_limit=args.similarity_limit,
            index_name="search-products-stage-cz", batch_size=batch_size,
        )
        unformatted_candidates_by_source[source] = unformatted_candidates
        # {source: {offer_id: [product_id, ...]}}
        candidates_by_source[source] = candidate_provider._format_candidates(unformatted_candidates)

    # merge unformatted candidates
    merged_unformatted_candidates = merge_unformatted_candidates(unformatted_candidates_by_source)

    # merge newly obtained candiates with those present in data (distinct offer sets)
    offer_id_to_product_ids = defaultdict(set)
    offer_products_sources = defaultdict(lambda: defaultdict(set))

    for offer_id, offer in offers_data.items():
        if sd_candidates_source := offer.get("candidates_sources"):
            for source, candidates in sd_candidates_source.items():
                if source in args.candidates_sources:
                    offer_id_to_product_ids[offer_id] |= set(candidates)
                    for candidate in candidates:
                        offer_products_sources[offer_id][candidate].add(source)

    for source, offer_to_candidates in candidates_by_source.items():
        for offer_id, candidates in offer_to_candidates.items():
            if len(candidates) == 0:
                continue

            offer_id_to_product_ids[offer_id] |= set(candidates)
            for candidate in candidates:
                # mark the candidate source
                offer_products_sources[offer_id][candidate].add(source)

    return merged_unformatted_candidates, offer_id_to_product_ids, offer_products_sources


async def _get_data_and_candidates(args, offer_ids_to_pair, batch_size):
    remote_services = await get_remote_services(['cs2', 'elastic'])
    downloader = CS2Downloader(remote_services)

    # just don't download new full data for faiss, it might take eternity
    faiss_downloader_params = None
    build_index = False
    if FAISS_CANDIDATES_SOURCE in args.candidates_sources:
        build_index = determine_build_index(args.input_collector_offers)

    # get candidate providers, load fasttext model only if faiss is used
    candidate_provider = await get_candidate_provider(
        args, args.input_fasttext, faiss_downloader_params, remote_services, build_index
    )

    offers_data, candidate_search_info = await get_offers_data(args, offer_ids_to_pair, downloader)

    unformatted_candidates, offer_id_to_product_ids, offer_products_sources = await get_candidates(
        args, offers_data, candidate_search_info, candidate_provider, batch_size
    )

    # download data for candidate products and original pairing
    # some products' data might be retrieved multiple times in case the product is a candidate in separate processes
    products_data = await get_products_data(args, offers_data, offer_id_to_product_ids, downloader)

    offers_to_pop = []
    if args.remove_longtail_candidates:
        for offer_id, product_ids in offer_id_to_product_ids.items():
            before_long_tail_removal = len(product_ids)
            offer_id_to_product_ids[offer_id] = [pid for pid in product_ids if not products_data[pid].get("long_tail", 0)]
            n_product_ids = len(offer_id_to_product_ids[offer_id])

            if before_long_tail_removal != n_product_ids:
                logging.info(f"Removed {before_long_tail_removal - n_product_ids} longtail candidates from offer {offer_id}")
            # remove offer if if no products left (theoretically it could happen)

            if n_product_ids == 0:
                offers_to_pop.append(offer_id)

    for offer_id in offers_to_pop:
        _ = offer_id_to_product_ids.pop(offer_id)

    await remote_services.close_all()

    return dict(offer_id_to_product_ids), offers_data, products_data, dict(offer_products_sources), dict(unformatted_candidates)


def get_data_and_candidates(args, offer_ids_to_pair, batch_size):
    return asyncio.run(_get_data_and_candidates(args, offer_ids_to_pair, batch_size))


async def get_matches(
    offer_id,
    offer_data: dict,
    product_ids: list,
    products_data: dict,
    products_sources: dict,
    xgboostmodel: XGBoostMatchingModel,
    prioritize_status: bool = True,
    prioritize_name_match: bool = True,
):
    match_yes = []
    match_no = []
    match_unknown = []
    invalid_parsing = []
    invalid_offer_data = {}

    try:
        parsed_offer = Offer.parse_obj(offer_data)
        # match_name is used in matchapi and during training
        if offer_data["match_name"]:
            parsed_offer.name = offer_data["match_name"]
    except Exception as e:
        logging.info(f'Parsing offer {offer_id} for matching failed - no results, {e}')
        invalid_offer_data = {
                'decision': 'invalid offer data',
                'item_id': offer_id,
                'exception': str(e),
                'data': str(offer_data),
                'confidence': None
            }
        return Matches(match_yes, match_no, match_unknown, invalid_parsing, invalid_offer_data)

    for product_id in product_ids:
        product_data = products_data.get(product_id, {})
        try:
            product_data["attributes"] = rinse_product_attributes(
                offer_id, product_data.get("attributes", {}), product_data.get("offers_attributes", {})
            )
            parsed_product = Product.parse_obj(product_data)
        except ValidationError as e:
            # parsing failed
            invalid_parsing.append({
                'decision': 'invalid data',
                'item_id': offer_id,
                'candidate_id': product_id,
                'details': f"data: {str(product_data)}",
                'sources': ','.join(sorted(list(products_sources[product_id]))),
                'confidence': None,
                'candidate': {"data": {}},

            })
            logging.info(f'Parsing product {product_id} for matching failed: {e}')
            continue
        try:
            match = await xgboostmodel(parsed_product, parsed_offer)
        except Exception as e:
            logging.exception(f"Exception while predicting for offer {offer_id} and product {product_id}")
            invalid_parsing.append({
                'decision': 'failed prediction',
                'item_id': offer_id,
                'candidate_id': product_id,
                'details': f"data: {e.__str__}",
                'sources': ','.join(sorted(list(products_sources[product_id]))),
                'confidence': None,
                'candidate': {"data": {}},

            })
            continue
        # get normalized product and offer names
        norm_product_name = xgboostmodel.namesimilarity.pipeline(parsed_product.name)
        norm_offer_name = xgboostmodel.namesimilarity.pipeline(parsed_offer.name)

        decision = match.match
        res = {
            'decision': decision.value,
            'item_id': offer_id,
            'candidate_id': product_id,
            'details': match.details,
            'confidence': match.confidence,
            'sources': ','.join(sorted(list(products_sources[product_id]))),
            'norm_product_name': norm_product_name,
            'norm_offer_name': norm_offer_name,
            # parse product status as it would be in candy
            'candidate': {"data": {"status": {"id": product_data["status"]}}} if "status" in product_data else {"data": {}},
        }
        if decision == Decision.yes:
            match_yes.append(res)
        elif decision == Decision.no:
            match_no.append(res)
        elif decision == Decision.unknown:
            match_unknown.append(res)

    # prioritize products by status hierarchy and name match
    additional_unknown = []
    if len(match_yes) > 0:
        if prioritize_status:
            # run with feature_oc = False
            match_yes, match_no, match_unknown = Candy._prioritize_status(match_yes, match_no, match_unknown, False)
        if prioritize_name_match:
            match_yes, additional_unknown = Candy._prioritize_name_match(match_yes)

    matches = Matches(match_yes, match_no, match_unknown + additional_unknown, invalid_parsing, invalid_offer_data)

    return matches


async def run_matching(args, offer_ids_to_pair: list, mode: str = "validation", preceding_item_ids: List[str] = []):
    assert mode in ["validation", "tuning"], "Incorrect mode, choose from 'validation' and 'tuning'"
    n_items = len(offer_ids_to_pair)

    batch_size = 15
    n_processes = min(4, ceil(n_items / 15))
    logging.info(f"Getting candidates and data in {n_processes} processes")

    n = len(offer_ids_to_pair)
    chunks = [offer_ids_to_pair[int(n*(i/n_processes)):int(n*((i + 1)/n_processes))] for i in range(n_processes)]
    args_list = [(args, chunk, batch_size) for chunk in chunks]

    with Pool(n_processes) as pool:
        results = pool.starmap(get_data_and_candidates, args_list)

    # id to candidates ids
    offer_id_to_product_ids = {}
    # pure data
    offers_data = {}
    # data for candidates
    products_data = {}
    # offer to sources of candidates
    offer_products_sources = {}

    unformatted_candidates = {}
    for r in results:
        offer_id_to_product_ids.update(r[0])
        offers_data.update(r[1])
        products_data.update(r[2])
        offer_products_sources.update(r[3])
        unformatted_candidates.update(r[4])

    # monitoring only newly obtained candidates
    candidates_monitor = CandidatesMonitor(args.candidates_sources)
    for item_id, candidates_dict in unformatted_candidates.items():
        id_paired = offers_data.get(item_id, {}).get("product_id")
        item_candidates = list(candidates_dict.values())
        candidates_monitor.monitor_incoming_candidates(
            item_candidates, int(item_id), id_paired
        )

    # no candidate offers
    no_candidate_offers = []
    for offer_id, offer_data in offers_data.items():
        if offer_id not in offer_id_to_product_ids:
            no_candidate_offers.append(offer_data)

    xgboostmodel = get_matching_model(args)

    # get matching results
    n_to_match = len(offer_id_to_product_ids.items())
    cnt = 0
    final_decisions_data = defaultdict(list)
    invalid_offers = []
    # store offers matched to same product as currently are
    paired_product_is_matched = []

    for offer_id, product_ids in offer_id_to_product_ids.items():
        cnt += 1
        if cnt % 50 == 0:
            logging.info(f"{cnt}/{n_to_match}")
        offer_data = offers_data[offer_id]
        Match = await get_matches(
            offer_id, offer_data, product_ids, products_data, offer_products_sources[offer_id],
            xgboostmodel, args.prioritize_status, args.prioritize_name_match
        )

        # assign the matches according to final decision
        full_matches = Match.yes + Match.no + Match.unknown + Match.invalid
        invalid_offer = Match.invalid_offer
        if invalid_offer:
            invalid_offers.append(invalid_offer)
        elif len(Match.yes) == 1:  # exactly one match
            final_decisions_data['matched'].append(full_matches)
            if int(Match.yes[0]["candidate_id"]) == int(offer_data["product_id"]):
                paired_product_is_matched.append(full_matches)
        elif len(Match.yes) == 0 and len(Match.unknown) == 0:  # no matches, create new product
            final_decisions_data['new_product'].append(full_matches)
        elif len(Match.yes) > 1 or (len(Match.yes) == 0 and len(Match.unknown) > 0):  # inconclusive
            final_decisions_data['unknown'].append(full_matches)

        # not clear what to do
        # elif (len(Match.yes) == 0 and len(Match.unknown) == 0 and len(Match.invalid) > 0):
        #    ...

    # upper threshold finetuning or validation with set thresholds
    if mode == "tuning":
        # set thresholds to try
        thresholds = [i/200 for i in range(60, 200, 1)]

        thr_results = pd.DataFrame(np.nan, index=range(len(thresholds)), columns=["threshold", "precision", "matched_pct"])

        # recalculate results for given threshold from current decisions and calculate precision and coverage
        for i, thr in enumerate(thresholds):
            precision_thr, _, matched_pct = rematch(thr, final_decisions_data, offers_data, n_items, args.prioritize_status)
            thr_results.iloc[i, :] = [thr, precision_thr, matched_pct]

        # save threshold results
        thr_results_path = str(Path(args.data_directory) / "thr_results.csv")
        thr_results.to_csv(thr_results_path, index=False)
        mlflow.log_artifact(thr_results_path)

        thresholds_evaluator = ThresholdsEvaluator(thr_results)

        # visualize the results
        thresholds_plot_path = str(Path(args.data_directory) / "thr_results_plot.png")
        thresholds_evaluator.plot_threshold_results(thresholds_plot_path)
        mlflow.log_artifact(thresholds_plot_path)

        # get best scores for multiple possible weighing of precision and matched_pct
        top_scores_for_weights = thresholds_evaluator.get_weight_maximum_scores()
        tsfw_path = str(Path(args.data_directory) / "top_scores_for_weights.csv")
        top_scores_for_weights.to_csv(tsfw_path, index=False)
        mlflow.log_artifact(tsfw_path)

        # get number one score, the one that won for highest number of possible weight combinations
        top_threshold = thresholds_evaluator.get_top_threshold(top_scores_for_weights)
        logging.info(f"Finetuned threshold: {top_threshold}")
        # just to keep current format and not create breaking change
        thresholds = (top_threshold, top_threshold)
    else:
        # validation mode
        thresholds = xgboostmodel.thresholds
        # validation, create excel
        logging.info('FINAL MATCHES COUNTS')
        for k, v in final_decisions_data.items():
            logging.info(f'{k}: ')
            logging.info(len(v))
            # log to mlflow
            mlflow.log_metric(k, len(v))
        logging.info(f"No candidate found: {len(no_candidate_offers)}")
        mlflow.log_metric("no_candidate_found", len(no_candidate_offers))

        # overall results
        per_category_results = log_results(
            args,
            final_decisions_data,
            paired_product_is_matched,
            n_items,
            offers_data,
            True,
            None,
            "Overall",
        )
        if preceding_item_ids:
            # retraining, report also metrics for preceding and new items separately
            preceding_ids = list(set(offer_id_to_product_ids.keys()).intersection(set(preceding_item_ids)))
            _ = log_results(
                args,
                final_decisions_data,
                paired_product_is_matched,
                n_items,
                offers_data,
                True,
                preceding_ids,
                "Preceding items"
            )
            new_item_ids = list(set(offer_id_to_product_ids.keys()) - set(preceding_item_ids))
            _ = log_results(
                args,
                final_decisions_data,
                paired_product_is_matched,
                n_items,
                offers_data,
                True,
                new_item_ids,
                "New items"
            )

        # candidates monitoring
        candidates_monitor.produce_candidate_statistics(create_plot=True)
        candidates_monitor.log_metrics_mlflow()
        pdf_candidate_monitoring = candidates_monitor.pdf_source_statistics

        # write and log the final excel file
        docwriter = DocumentCreator(offers_data, products_data)

        no_candidate_sheet_data = {'no_candidates': pd.DataFrame(
            [docwriter.create_item_part_of_row(offer) for offer in no_candidate_offers]
        )}

        output_path_sheet = args.data_directory + '/matching_results.xlsx'
        docwriter.create_final_excel(
            output_path_sheet,
            final_decisions_data,
            no_candidate_sheet_data,
            {
                'candidates_metrics': pdf_candidate_monitoring,
                'invalid_offer_data': pd.DataFrame(invalid_offers),
                'per_category_results': per_category_results
            },
        )
        mlflow.log_artifact(output_path_sheet)

        distance_dist_fig = candidates_monitor.results['similarity_plot']
        if distance_dist_fig:
            output_path_fig = args.data_directory + '/distance_distributions.png'
            distance_dist_fig.savefig(output_path_fig, dpi=200)
            mlflow.log_artifact(output_path_fig)

    # write thresholds to file and log it
    # either finetuned thresholds or default ones
    thresholds_path = str(Path(args.data_directory) / "thresholds.txt")
    with open(thresholds_path, "w") as f:
        f.write(
            ",".join(
                (str(thr) for thr in thresholds)
            )
        )
    return thresholds_path


@notify
def main(args):
    all_ids_to_pair = [str(id) for id in read_lines_with_int(args.test_items_ids_file)]

    if args.preceding_test_items_ids_file:
        preceding_offer_ids_to_pair = [str(id) for id in read_lines_with_int(args.preceding_test_items_ids_file)]
        logging.info(f"Loaded {len(preceding_offer_ids_to_pair)} preceding items ids")

        new_offer_ids_to_pair = list(set(all_ids_to_pair) - set(preceding_offer_ids_to_pair))

        all_ids_to_pair = preceding_offer_ids_to_pair + new_offer_ids_to_pair
        # mark source of item id to split them so that both new and preceding offers are present in both finetuning and validation list
        stratify_ar = ["preceding"]*len(preceding_offer_ids_to_pair) + ["created"]*len(new_offer_ids_to_pair)
    else:
        # constant
        stratify_ar = ["created"]*len(all_ids_to_pair)
        # used for filtering ids
        preceding_offer_ids_to_pair = []

    if str2bool(args.finetune_thresholds):
        n_items_to_finetune = round(len(all_ids_to_pair)/2)
        # split
        offer_ids_to_finetune, offer_ids_to_validate = train_test_split(
            all_ids_to_pair, train_size=n_items_to_finetune, random_state=10, stratify=stratify_ar
        )

        logging.warning(f"Finetuning threshold on {n_items_to_finetune} items")
        # run threshold tuning
        args.thresholds_path = asyncio.run(run_matching(args, offer_ids_to_finetune, "tuning"))
    else:
        offer_ids_to_validate = all_ids_to_pair

    logging.info(f"Evaluating on {len(offer_ids_to_validate)} items")
    # run either with default thresholds (no finetuning) or with saved finetuned thresholds
    args.thresholds_path = asyncio.run(run_matching(args, offer_ids_to_validate, "validation", preceding_offer_ids_to_pair))
    # log thresholds
    mlflow.log_artifact(args.thresholds_path)


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), force=True)
    parser = argparse.ArgumentParser()

    parser.add_argument("--categories", required=True)
    parser.add_argument("--input-collector", default=None)
    parser.add_argument("--use-collector-data", type=str2bool, default=False)
    parser.add_argument("--input-attributes", required=True)
    parser.add_argument("--input-fasttext", required=True)
    parser.add_argument("--input-xgb", required=True)
    parser.add_argument("--data-directory", default="/data")
    parser.add_argument("--similarity-limit", default=1)
    parser.add_argument("--max-candidates", default=10)
    parser.add_argument("--candidates-sources", default=FAISS_CANDIDATES_SOURCE)
    parser.add_argument("--finetune-thresholds", default="true")
    parser.add_argument("--thresholds-path", default=None)
    parser.add_argument("--prioritize-status", type=str2bool, default=True)
    parser.add_argument("--prioritize-name-match", type=str2bool, default=True)
    parser.add_argument("--remove-longtail-candidates", type=str2bool, default=True)
    parser.add_argument("--test-items-ids-file", type=str_or_none)
    parser.add_argument("--test-items-data-file", type=str_or_none)
    parser.add_argument("--preceding-test-items-ids-file", default=None, type=str_or_none)
    parser.add_argument("--preceding-test-items-data-file", default=None, type=str_or_none)
    parser.add_argument("--preceding-data-directory", default="/preceding_data")
    parser.add_argument("--unit-conversions", default=True, type=str2bool)
    parser.add_argument("--price-reject-a", default=1000.0, type=float)
    parser.add_argument("--price-reject-b", default=400.0, type=float)
    parser.add_argument("--price-reject-c", default=2.5, type=float)
    parser.add_argument("--tok-norm-args", required=False)  # @@@ separated key=value pairs
    parser.add_argument("--matched-confidence-threshold", default=.95, type=float)  # currently only at this place
    parser.add_argument("--precision-confidence-threshold", default=.95, type=float)  # currently only at this place
    # used to estimate a probability of new model trained on the same dataset being better than the one used to produce
    # the provided per-category results
    parser.add_argument("--per-category-results-to-compare", default=None, type=str_or_none)

    args = parser.parse_args()

    args.input_collector = merge_collector_folders(
        process_inputs(args.input_collector.split("@"), args.data_directory), args.data_directory
    )
    args.input_collector_products = args.input_collector + "/products"
    args.input_collector_offers = args.input_collector + "/offers"
    args.input_attributes = process_input(args.input_attributes, args.data_directory)
    args.input_fasttext = process_input(args.input_fasttext, args.data_directory)
    args.input_xgb = process_input(args.input_xgb, args.data_directory)

    args.test_items_ids_file = process_input(args.test_items_ids_file, args.data_directory)
    args.test_items_data_file = process_input(args.test_items_data_file, args.data_directory)

    # there should be exactly one file present
    if args.test_items_data_file and os.path.isdir(args.test_items_data_file):
        assert len(os.listdir(args.test_items_data_file)) == 1
        args.test_items_data_file = os.path.join(
            args.test_items_data_file,
            os.listdir(args.test_items_data_file)[0]
        )

    args.preceding_test_items_ids_file = process_input(args.preceding_test_items_ids_file, args.preceding_data_directory)
    args.preceding_test_items_data_file = process_input(args.preceding_test_items_data_file, args.preceding_data_directory)
    # there should be exactly one file present
    if args.preceding_test_items_data_file and os.path.isdir(args.preceding_test_items_data_file):
        assert len(os.listdir(args.preceding_test_items_data_file)) == 1
        args.preceding_test_items_data_file = os.path.join(
            args.preceding_test_items_data_file,
            os.listdir(args.preceding_test_items_data_file)[0]
        )

    args.candidates_sources = args.candidates_sources.replace(' ', '').split(',')

    args.per_category_results_to_compare = process_input(args.per_category_results_to_compare, args.data_directory)

    args = set_at_args(args, 'tok_norm_args', args.data_directory)

    logging.info(args)

    with mlflow.start_run():
        main(args)
