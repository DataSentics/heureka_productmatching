import os
import mlflow
import argparse
import logging
import time

from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
from itertools import chain

from utilities.component import process_input, process_inputs, compress
from utilities.normalize import normalize_string
from utilities.normalize.regexes import get_regexes
from utilities.loader import Product, Corpus, merge_collector_folders
from utilities.notify import notify
from utilities.args import str_or_none
from utilities.logger_to_file import log_to_file_and_terminal

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), force=True)


@notify
def save_titles(args):
    products_path = args.input_collector + "/products"
    offers_path = args.input_collector + "/offers"

    def get_titles_product(input_product, offers_path, regexes):
        index, product = input_product
        titles = set()

        titles.add(normalize_string(string=product["name"], regexes=regexes))

        for offer in Product.offers(offers_path, product["id"]):
            titles.add(normalize_string(string=offer["name"], regexes=regexes))

            if offer["offer_name"] is not None:
                titles.add(normalize_string(string=offer["offer_name"], regexes=regexes))

        if index % 10_000 == 0:
            logging.info(f"Parsed {index} index (unordered).")

        return titles

    start_time = time.time()

    regexes, _ = get_regexes()

    pool = ThreadPool(cpu_count())
    titles_list = pool.map(lambda input_product: get_titles_product(input_product, offers_path, regexes),
                           Product.index_products(products_path))
    time_elapsed = time.time() - start_time
    titles = set(chain(*titles_list))

    logging.info("Collector data fully parsed.")
    logging.info(f"Parsed {len(titles_list)} products, {len(titles)} titles, in {round(time_elapsed, 2)}s" +
                 f" - {round(len(titles_list) / time_elapsed, 2)} products/s on average.")
    mlflow.log_metric("title_count", 0, step=0)
    mlflow.log_metric("title_count", len(titles), step=len(titles_list))

    # retraining mode, add titles from preceding corpus file
    if args.preceding_corpus:
        preceding_corpus_path = args.preceding_corpus + "/corpus.txt"
        n_titles_before = len(titles)

        with open(preceding_corpus_path, "r") as f:
            for sub_index, line in enumerate(f):
                index = len(titles_list) + sub_index
                title = line.replace("\n", "")

                titles.add(title)

                if index % 10_000 == 0:
                    logging.info(f"Parsed {index} index. {len(titles)} titles.")
                    mlflow.log_metric("title_count", len(titles), step=index)

        n_titles_added = len(titles) - n_titles_before
        logging.info(f"Added {n_titles_added} from file {preceding_corpus_path}. ")
        mlflow.log_metric("title_count", len(titles), step=index)

    output_path = args.data_directory + "/embedding_dataset/corpus.txt"

    for title in titles:
        Corpus.save(output_path, title)

    Corpus.close()

    tar_file = args.data_directory + "/embedding_dataset.tar.gz"
    compress(tar_file, args.data_directory + "/embedding_dataset")
    mlflow.log_artifact(tar_file)

    return {"n_titles": len(titles)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-collector", required=True)
    parser.add_argument("--preceding-corpus", default=None, type=str_or_none)
    parser.add_argument("--data-directory", default="/data")

    args = parser.parse_args()

    args.input_collector = merge_collector_folders(
        process_inputs(args.input_collector.split("@"), args.data_directory), args.data_directory
    )
    args.preceding_corpus = process_input(args.preceding_corpus, args.data_directory)

    with mlflow.start_run():
        log_to_file_and_terminal(save_titles, args)
