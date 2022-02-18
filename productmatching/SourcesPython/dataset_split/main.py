import os
import mlflow
import random
import argparse
import logging

from collections import Counter

from utilities.component import process_input, process_inputs, compress
from utilities.loader import Product, write_lines, read_lines_with_int, merge_collector_folders, load_json
from utilities.notify import notify
from utilities.args import str_or_none
from utilities.logger_to_file import log_to_file_and_terminal

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), force=True)


@notify
def separate_test_items(args):
    products_path = args.input_collector + "/products"
    offers_path = args.input_collector + "/offers"

    # delete offers for products with wrong status
    # they will not interfere with anything else afterwards, but we won't lose the data
    cat_2_wrong_products = Product.categories_to_products_without_status(
        os.environ.get("PRODUCT_STATUSES", "11").split(','),
        products_path,
        args.categories
    )

    logging.info('Deleting offer files for products with wrong status.')
    for cat, product_ids in cat_2_wrong_products.items():
        for product_id in product_ids:
            Product.delete_offers(offers_path, products_path, product_id)

    # get product in categories, info about counts
    categories_to_products = Product.categories_to_products(products_path, categories=args.categories)
    pr_to_of = {}
    for cat, product_ids in categories_to_products.items():
        n_offers_cat = 0
        for product_id in product_ids:
            pr_to_of[product_id] = Product.n_offers(offers_path, product_id)
            n_offers_cat += pr_to_of[product_id]
        logging.info(f"N products {cat}: {len(product_ids)}")
        logging.info(f"N offers {cat}: {n_offers_cat}")

    test_size = 1 - args.train_size

    random.seed(123)

    test_items_data = []
    test_items_ids = []
    for cat, product_ids in categories_to_products.items():
        n_cat_test_items_ids = 0
        # number of offers in category
        n_offers = sum((Product.n_offers(offers_path, product_id) for product_id in product_ids))
        # either percetage of offers or minimal count
        n_test_items = min(int(n_offers * test_size), args.max_test_items_in_category)
        # randomly select products to select testing offers
        repeated_product_ids = [
            pr
            for pr in product_ids
            for _ in range(pr_to_of[pr])
        ]
        test_items_products = random.sample(repeated_product_ids, k=n_test_items)
        # number of offers per test product
        product_to_n_test_offers = Counter(test_items_products)

        for product_id, n_test_offers in product_to_n_test_offers.items():
            # get offers
            offers = Product.offers(offers_path, product_id)
            # select testing offers
            test_offers = random.sample(offers, k=n_test_offers)
            test_items_data.extend(test_offers)
            test_items_ids.extend(o["id"] for o in test_offers)

            n_cat_test_items_ids += len(test_offers)

            # remove offers of selected product from training data
            # to avoid getting similar offers both in training and testing data
            Product.delete_offers(offers_path, products_path, product_id)

        mlflow.log_metric(f"{cat}_selected_n_test_items", n_cat_test_items_ids)
        logging.info(f"{cat}_selected_n_test_items {n_cat_test_items_ids}")

    if args.preceding_test_items_ids_file:
        preceding_test_items_ids = read_lines_with_int(args.preceding_test_items_ids_file)
        logging.info(f"n_preceding_test_items: {len(preceding_test_items_ids)}")
        mlflow.log_metric("n_preceding_test_items", len(preceding_test_items_ids))

        # get offers to products connection for testing offers
        product_to_offers = Product.get_products_for_offers(offers_path, preceding_test_items_ids)

        for product_id in product_to_offers:
            Product.delete_offers(offers_path, products_path, product_id)

        test_items_ids.extend(preceding_test_items_ids)

    # info about counts
    categories_to_products = Product.categories_to_products(products_path, args.categories)
    for cat, pr in categories_to_products.items():
        logging.info(f"{cat}_n_products: {len(pr)}")
        logging.info(f"{cat}_total_n_train_offers: {sum([Product.n_offers(offers_path, p) for p in pr])}")
        mlflow.log_metric(f"{cat}_n_products", len(pr))
        mlflow.log_metric(f"{cat}_total_n_train_offers", sum([Product.n_offers(offers_path, p) for p in pr]))

    # write test items to file
    test_items_ids_path = os.path.join(args.data_directory, "test_items.list")
    write_lines(test_items_ids_path, test_items_ids)
    mlflow.log_artifact(test_items_ids_path)
    mlflow.log_metric("overall_n_test_items", len(test_items_ids))

    test_items_data_dir = os.path.join(args.data_directory, "test_items_data")
    os.makedirs(test_items_data_dir, exist_ok=True)
    test_items_data_path = os.path.join(test_items_data_dir, "test_items_data.txt")
    write_lines(test_items_data_path, test_items_data)
    tar_file = os.path.join(args.data_directory, "test_items_data.tar.gz")
    compress(tar_file, test_items_data_dir)
    mlflow.log_artifact(str(tar_file))

    output_collector_dirname = "train_collector_data"
    # rename dir to keep the structure and be able to use it further in flow
    new_dir_name = os.path.join(
        os.path.dirname(args.input_collector),
        output_collector_dirname
    )

    os.rename(args.input_collector, new_dir_name)

    # log train collector data
    tar_file = os.path.join(args.data_directory, f"{output_collector_dirname}.tar.gz")
    compress(tar_file, new_dir_name)
    mlflow.log_artifact(str(tar_file))


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), force=True)

    parser = argparse.ArgumentParser()

    parser.add_argument("--input-collector", required=True)
    parser.add_argument("--categories", required=True)
    parser.add_argument("--preceding-test-items-ids-file", default=None, type=str_or_none)
    parser.add_argument("--data-directory", default="/data")
    parser.add_argument("--max-test-items-in-category", type=int, default=750)
    parser.add_argument("--train-size", type=float, default=0.9)

    args = parser.parse_args()

    args.input_collector = merge_collector_folders(
        process_inputs(args.input_collector.split("@"), args.data_directory), args.data_directory
    )
    args.categories = ",".join(sorted(c.strip() for c in args.categories.split(",")))

    args.preceding_test_items_ids_file = process_input(args.preceding_test_items_ids_file, args.data_directory)

    logging.info(args)

    with mlflow.start_run():
        log_to_file_and_terminal(separate_test_items, args)
