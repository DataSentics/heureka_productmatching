import argparse
import asyncio
import json
import logging
import os
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import mlflow

from utilities.args import str2bool
from utilities.component import process_inputs, compress
from utilities.loader import Corpus, Product, merge_collector_folders, safe_directory
from utilities.notify import notify
from utilities.s3_utils import download_from_s3, upload_to_s3


def update_s3_info(args):
    _, updated_sd_info = get_updated_s3_info(args)
    info_path = os.path.join(args.data_directory, args.datasets_info_file)
    with open(info_path, "w") as fw:
        json.dump(updated_sd_info, fw)

    if args.upload_to_s3:
        s3_info_path = f"{args.datasets_s3_bucket}/{args.datasets_info_file}"
        logging.info("Uploading updated info file.")
        mlflow.log_param("updated_datasets_info_file", str(updated_sd_info))
        upload_to_s3(s3_info_path, info_path)
        os.remove(info_path)


def get_updated_s3_info(args):
    # {"1962": {"versions": {1: "2021-10-06", 2: "2021-10-07"}}, "1963": ...}
    s3_path = f"{args.datasets_s3_bucket}/{args.datasets_info_file}"

    local_path = download_from_s3(s3_path, args.data_directory)
    with open(local_path, "r") as fr:
        sd_info = json.load(fr)

    os.remove(local_path)

    # the empty info file contains "{}"
    sd_info = defaultdict(dict, sd_info)

    categories = args.categories.split(",")
    if args.datasets_to_create:
        categories = args.datasets_to_create

    category_dataset_versions = {}

    for category in categories:
        if sd_category_info := sd_info.get(category):
            dataset_version = str(max(int(k) for k in sd_category_info["versions"].keys()) + 1)
            logging.warning(f"Static dataset for provided category {category} already exists, creating new version v{dataset_version} of the dataset.")
        else:
            dataset_version = "0"
            sd_info[category]["versions"] = {}
            logging.info(f"Setting new static dataset id for category: {category}, creating version v0")

        sd_info[category]["versions"][dataset_version] = str(datetime.now())
        category_dataset_versions[category] = dataset_version

    return category_dataset_versions, sd_info


def split_products_offer_files(args):
    sd_category_directories = {}
    offers_files = os.listdir(args.input_collector_offers)
    for product in Product.products_for_categories(args.input_collector_products, args.datasets_to_create):
        category = product["category_id"]
        cat_folder = os.path.join(args.data_directory, f"static_dataset_{category}")
        sd_category_directories[str(category)] = cat_folder
        Corpus.save(
            os.path.join(cat_folder, "products", "products.txt"),
            json.dumps(product)
        )
        offer_file = f"product.{product['id']}.txt"
        if offer_file in offers_files:
            cat_offers_folder = os.path.join(cat_folder, "offers")
            safe_directory(cat_offers_folder)
            shutil.move(
                os.path.join(args.input_collector_offers, offer_file),
                os.path.join(cat_offers_folder, offer_file)
            )

    Corpus.close()

    # add info about missing categories which should be created
    missing_categories = list(set(args.datasets_to_create) - set(sd_category_directories.keys()))
    # safety check, raise error if >=80% of categories is missing, it usually means an error or mistake
    if len(missing_categories) > len(args.datasets_to_create)*0.8:
        raise ValueError(f"{len(missing_categories)}/{len(args.datasets_to_create)} missing during creating static dataset, check the process and category ids. Missing categories {missing_categories}")

    for missing_cat in missing_categories:
        logging.warning(f"Creating empty directory for missing category {missing_cat}")
        folder = os.path.join(args.data_directory, f"static_dataset_{missing_cat}")
        os.mkdir(folder)
        sd_category_directories[str(missing_cat)] = folder

    return sd_category_directories


async def main(args):
    category_dataset_versions, _ = get_updated_s3_info(args)
    print(category_dataset_versions)

    mlflow.log_param("category_dataset_versions", str(category_dataset_versions))
    logging.info(f"creating dataset for categories with versions: {category_dataset_versions}")

    sd_category_directories = split_products_offer_files(args)

    for category, cat_folder in sd_category_directories.items():
        tar_file = f"{cat_folder}.tar.gz"
        compress(tar_file, cat_folder)

        if args.upload_to_s3:
            dataset_version = category_dataset_versions[category]
            static_dataset_s3_path = f"{args.datasets_s3_bucket}/{category}/v{dataset_version}/static_dataset_{category}.tar.gz"
            logging.info(f"Uploading dataset to {static_dataset_s3_path}.")
            upload_to_s3(static_dataset_s3_path, tar_file)
            mlflow.log_param(f"static_dataset_path_{category}", static_dataset_s3_path)

    update_s3_info(args)


@notify
def create_static_dataset(args):
    asyncio.run(main(args))


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), force=True)
    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets-to-create", required=True)
    parser.add_argument("--categories", required=True)
    parser.add_argument("--input-collector", default=None)
    parser.add_argument("--data-directory", default="/data")
    parser.add_argument("--datasets-info-file", default="category_dataset_info.json")
    parser.add_argument("--datasets-s3-bucket", default="s3://ml-static-datasets")
    parser.add_argument("--upload-to-s3", type=str2bool, default="true")

    args = parser.parse_args()

    args.datasets_to_create = args.datasets_to_create.split(',')
    args.categories = ",".join(sorted(c.strip() for c in args.categories.split(",")))
    args.input_collector = Path(merge_collector_folders(
        process_inputs(args.input_collector.split("@"), args.data_directory), args.data_directory
    ))
    args.input_collector_products = args.input_collector / "products"
    args.input_collector_offers = args.input_collector / "offers"

    args.data_directory = Path(args.data_directory)

    logging.info(args)

    with mlflow.start_run():
        create_static_dataset(args)
