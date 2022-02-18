import argparse
import asyncio
import logging
import os
import mlflow
import pandas as pd

from utilities.notify import notify
from utilities.component import compress
from utilities.cs2_downloader import CS2Downloader
from utilities.remote_services import get_remote_services


async def download_products_batches(category, args):
    """
    Downlaod from given cateogry given number of products from given starting id. 
    Parse them to dataframe for easier manipulation.
    """
    remote_services = await get_remote_services(['cs2'])
    downloader = CS2Downloader(remote_services)

    fields = ["id", "category_id", "name",  "slug", "category.slug", "status", "prices"]
    max_products = args.max_products_in_category
    limit = args.batch_limit

    # downlaod from given category given number of products
    products = []
    async for products_batch in downloader.products_download_range(
        category, fields=fields, max_products=max_products, limit=limit
    ):
        for product in products_batch:
            product["category_slug"] = product.get("category", {}).get("slug", "")
        products.extend(products_batch)

    # parse them to dataframe for easier manipulation
    products_df = pd.DataFrame(products)
    await remote_services.close_all()

    return products_df


async def download_products(args):
    remote_services = await get_remote_services(['cs2'])
    downloader = CS2Downloader(remote_services)
    coros = []
    products_df = pd.DataFrame()
    for category in args.categories:
        logging.info(f"Getting info about category {category}")
        # get number of products in given category
        response = await downloader.category_info(category, ["product_count"])
        if not response:
            logging.info(f"NO info about category {category}")
            continue
        n_products = response[0]["product_count"]

        if not n_products:
            logging.info(f"Skipping category {category}, no products.")
            continue
        # create individual downloading coroutines fro each category
        coros.append(download_products_batches(category, args))

    category_products_dfs = await asyncio.gather(*coros)
    # parse results
    products_df = pd.concat(category_products_dfs, ignore_index=True)

    logging.info(f"{len(products_df)} products donwloaded")
    mlflow.log_metric("n_products_downloaded", len(products_df))

    output_dir = os.path.join(args.data_directory, "clustering_data")
    os.makedirs(output_dir, exist_ok=True)

    # save data and log them to mlflow
    products_df.to_csv(os.path.join(output_dir, "clustering_data.csv"), index=False)

    tar_file = os.path.join(args.data_directory, "clustering_data.tar.gz")
    compress(tar_file, output_dir)

    mlflow.log_artifact(str(tar_file))

    await remote_services.close_all()


@notify
def main(args):
    asyncio.run(download_products(args))


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), force=True)
    parser = argparse.ArgumentParser()

    parser.add_argument("--categories", required=True)
    parser.add_argument("--max-products-in-category", required=True, type=int)
    parser.add_argument("--data-directory", default="/data")
    parser.add_argument("--batch-limit", default=200, type=int)

    args = parser.parse_args()

    args.categories = args.categories.replace(" ", "").split(",")

    with mlflow.start_run():
        main(args)
