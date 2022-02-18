import os
import argparse
import logging
import datetime
import pandas as pd
import typing as t

from preprocessing.models import exact_match
from utilities.attributes import (
    Attributes,
    get_collector_product_offers_attributes,
    parse_attributes,
    rinse_product_attributes
)
from utilities.loader import Product
from xgboostmatching.models import features

import mlflow

XGB_ID_INFO_COLS = ["query_product_id", "offer_id", "product_id"]
XGB_INFO_COLS = ["label", *XGB_ID_INFO_COLS]


def get_batch_indexes(job_no, n_items, n_jobs, max_len_batch=-1):
    min_index_fl = job_no * n_items / n_jobs
    min_index = int(min_index_fl)
    if max_len_batch > 0:
        max_index = min(int(min_index_fl + n_items / n_jobs), int(min_index_fl + max_len_batch / n_jobs))
    else:
        max_index = int(min_index_fl + n_items / n_jobs)

    return min_index, max_index


async def create_examples(
    args: argparse.Namespace,
    namesimilarity: exact_match.ExactMatchModel,
    attributes: Attributes,
    full_product_dict: t.Dict[str, dict],
    examples: pd.DataFrame,
    additional_product_data: t.DefaultDict[str, dict],
    selected_features: dict = None,
    log_mlflow: bool = True,
) -> pd.DataFrame:
    positive, negative = 0, 0

    # construct list of dicts with features, transform it to dataframe at the end
    examples_rows = []
    query_product_ids = examples["query_product_id"].unique()

    for product_index, query_product_id in enumerate(query_product_ids):

        logging.info(f'Product index {product_index}, positive {positive}, negative {negative} at {datetime.datetime.now().strftime("%H:%M")}')
        if log_mlflow and product_index % 1_00 == 0:
            mlflow.log_metric("negative_count", negative, step=product_index)
            mlflow.log_metric("positive_count", positive, step=product_index)
            mlflow.log_metric("product_index", product_index, step=product_index)

        query_product_examples = examples[examples["query_product_id"] == query_product_id]

        # common for all offers matched to this product
        offers = Product.offers(args.input_collector / "offers", query_product_id)

        query_product_offers = query_product_examples["offer_id"].unique()
        for offer_id in query_product_offers:
            query_product_offer_examples = query_product_examples[query_product_examples["offer_id"] == offer_id]

            # get info about offer
            try:
                offer = [of for of in offers if of["id"] == offer_id][0]
            except IndexError as e:
                print(query_product_id, type(offer_id), offer_id)
                raise e
            offer_attributes = parse_attributes(offer, field="attributes")
            offer_parsed_attributes = parse_attributes(offer, field="parsed_attributes")

            another_product_prices = [o["price"] for o in offers if o["id"] != offer["id"]]
            another_product_shops = [o["shop_id"] for o in offers if o["id"] != offer["id"]]
            another_product_eans = [o["ean"] for o in offers if o["id"] != offer["id"] and o["ean"]]

            for tup in query_product_offer_examples.itertuples(index=None):
                # get product info either from collector or downloaded data
                if "product_type" in tup._fields and tup.product_type == "product_downloaded":
                    product = additional_product_data[tup.product_id]
                else:
                    product = full_product_dict[str(tup.product_id)]

                # mapping of matched offers to attrributes and their values
                product_attributes_name_value_offers = get_collector_product_offers_attributes(args, product["id"])

                product_attributes = parse_attributes(product)

                if product["id"] != query_product_id:
                    # product is not 'query product' i.e. not paired to offer, use its data
                    prices = product["prices"]
                    shops = product["shops"]
                    eans = product["eans"]
                else:
                    # use info about product without the selected offer
                    prices = another_product_prices
                    shops = another_product_shops
                    eans = another_product_eans
                    product_attributes = rinse_product_attributes(offer_id, product_attributes, product_attributes_name_value_offers)

                features_list = await features.create(
                    product=features.Product(
                        name=product["name"],
                        prices=prices,
                        shops=shops,
                        category_id=product["category_id"],
                        attributes=product_attributes,
                        eans=eans,
                    ),
                    offer=features.Offer(
                        name=offer["match_name"],
                        price=offer["price"],
                        shop=offer["shop_id"],
                        attributes=offer_attributes,
                        parsed_attributes=offer_parsed_attributes,
                        ean=offer["ean"],
                    ),
                    namesimilarity=namesimilarity,
                    attributes=attributes,
                    names=True,
                    selected_features=selected_features,
                )

                examples_rows.append(
                    dict([
                        *zip(
                            XGB_INFO_COLS,
                            [tup.label, query_product_id, offer_id, product["id"]]
                        ),
                        *features_list
                    ])
                )

                if tup.label:
                    positive += 1
                else:
                    negative += 1

    # construct the final dataframe
    output_examples = pd.DataFrame.from_dict(examples_rows)
    created_features = [c for c in output_examples.columns if c not in XGB_INFO_COLS]

    logging.info(f"Created {positive} positive and {negative} negative samples.")
    logging.info(f"Created following basic features: {created_features}")
    if log_mlflow:
        mlflow.log_param("created_features", ",".join(created_features))
        mlflow.log_metric('positive_samples_total', positive)
        mlflow.log_metric('negative_samples_total', negative)

    return output_examples
