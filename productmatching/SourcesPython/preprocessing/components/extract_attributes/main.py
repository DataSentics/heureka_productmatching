import os
import json
import argparse
import logging
import ujson

import mlflow

from collections import defaultdict
from utilities.attributes import extract_offers_attributes
from utilities.component import process_input, process_inputs
from utilities.normalize import normalize_string
from utilities.loader import Product, Corpus, merge_collector_folders
from utilities.notify import notify
from utilities.args import str_or_none
from utilities.logger_to_file import log_to_file_and_terminal

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), force=True)


@notify
def extract(args):
    products_path = args.input_collector + "/products"
    offers_path = args.input_collector + "/offers"

    total_attributes = 0
    category_to_name_to_values = defaultdict(lambda: defaultdict(set))

    for step, product in enumerate(Product.products(products_path)):
        category = product["category_id"]

        if product["attributes"] is not None:
            for attribute in product["attributes"]:
                name = normalize_string(attribute["name"])
                value = normalize_string(attribute["value"])

                if "" in [name, value]:
                    continue

                category_to_name_to_values[category][name].add(value)

        if product["producers"] is not None:
            category_to_name_to_values[category]["producer"] |= {normalize_string(producer["name"]) for producer in product["producers"]}

        # add attributes from offers
        offers_attributes = extract_offers_attributes(Product.offers(offers_path, product_id=int(product["id"])))
        for name, values in offers_attributes.items():
            category_to_name_to_values[category][name] |= values

        if step % 10_000 == 0:
            logging.info(f"Parsed {step} step. {len(category_to_name_to_values[category])} attributes.")
            mlflow.log_metric("total_attributes", len(category_to_name_to_values[category]), step=step)

    # add atributes from preceding file
    if args.preceding_attributes:
        n_new_values = 0
        with open(args.preceding_attributes, "r") as attrfile:
            preceding_attributes = ujson.load(attrfile)

            for category, name_to_val in preceding_attributes.items():
                for name, values in name_to_val.items():

                    n_orig = len(category_to_name_to_values[int(category)][name])
                    category_to_name_to_values[int(category)][name] |= set([v[0] for v in values])
                    n_new_values += len(category_to_name_to_values[int(category)][name]) - n_orig

        logging.info(f"Added {n_new_values} new values from preceding attributes file {args.preceding_attributes}.")

    # creating list of lists of one words to mimic the structure of the external keywords file
    for category, name_to_val in category_to_name_to_values.items():
        for name, values in name_to_val.items():
            name_to_val[name] = sorted(list([[value] for value in values]))
        total_attributes += len(category_to_name_to_values[category])

    total_values = sum([
        len(val)
        for name_to_val in category_to_name_to_values.values()
        for name, val in name_to_val.items() if name != 'producer'
    ])
    mlflow.log_metric("total_values", total_values, step=step)
    mlflow.log_metric("total_attributes", total_attributes, step=step)

    output_path = args.data_directory + "/attributes.json"
    Corpus.write(output_path, json.dumps(category_to_name_to_values, indent=4, sort_keys=True))
    mlflow.log_artifact(output_path)
    logging.info('finished extract')

    return {"total_values": total_values, "total_attributes": total_attributes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-collector", required=True)
    parser.add_argument("--data-directory", default="/data")
    parser.add_argument("--preceding-attributes", default=None, type=str_or_none)

    args = parser.parse_args()

    args.input_collector = merge_collector_folders(
        process_inputs(args.input_collector.split("@"), args.data_directory), args.data_directory
    )
    args.preceding_attributes = process_input(args.preceding_attributes, args.data_directory)

    with mlflow.start_run():
        log_to_file_and_terminal(extract, args)
