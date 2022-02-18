import os
import json
import random
import logging
import shutil
from pathlib import Path
from os import listdir
from os.path import isfile, join
from typing import Optional, List, Union
from collections import defaultdict

PATH = os.path.dirname(os.path.realpath(__file__))


def safe_directory(path_in: Union[str, Path]):
    path = str(path_in)
    if "." in path:
        path = "/".join(path.split("/")[:-1])

    Path(path).mkdir(parents=True, exist_ok=True)


def count_lines(path: str):
    n_lines = sum(1 for line in open(path, "r", encoding="utf-8"))
    return n_lines


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.decoder.JSONDecodeError:
                logging.error(f'Cannot parse JSON line: {line}')


def load_line(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield line


def write_lines(path: str, lines: List):
    dump_lines = any(isinstance(l, dict) for l in lines)

    with open(path, "w") as f:
        if dump_lines:
            [f.write(f"{json.dumps(line)}\n") for line in lines]
        else:
            [f.write(f"{line}\n") for line in lines]


def remove_selected_lines(path: str, key_for_selection: str, lines_values_to_remove: List[int]):
    """
    Reads lines from file in 'path'. Each line is assumed to be a dict containing at least key used for filtering ('key_for_selection').
    The 'key_for_selection' is assumed to be integer-like. Function removes lines with key values contained in 'lines_values_to_remove' and saves the remaining lines to the same file.
    """
    orig_n_lines = count_lines(path)

    new_lines = [li for li in load_json(path) if int(li[key_for_selection]) not in lines_values_to_remove]
    new_n_lines = len(new_lines)

    if new_lines:
        # write remaining lines
        write_lines(path, new_lines)
        logging.info(f"Removed {orig_n_lines-new_n_lines} from {path}, {new_n_lines} lines remaining")
    else:
        # delete the file if no lines left
        os.remove(path)
        logging.info(f"Removed all {orig_n_lines-new_n_lines} from {path}, deleting the file")

    return len(new_lines)


def read_lines_with_int(path: str):
    """
    Reads lines from file, each line is assumed to contain one integer-like object.
    It is currently used for small files, one could use generator with yield clause with larger files.
    """
    with open(path, "r") as f:
        lines = [int(i.strip()) for i in f.readlines()]
    return lines


class Product:
    offer_files = None

    @classmethod
    def products(cls, path: Optional[str] = None):
        if not os.path.exists(path):
            raise KeyError(f"{path} does not exist")

        for root, dirs, files in os.walk(path):
            for file in files:
                if not file.endswith(".txt"):
                    continue

                for product in load_json(os.path.join(root, file)):
                    yield product

    @classmethod
    def products_for_categories(cls, path: Optional[str] = None, categories: List[str] = []):
        if not os.path.exists(path):
            raise KeyError(f"{path} does not exist")

        for root, dirs, files in os.walk(path):
            for file in files:
                if not file.endswith(".txt"):
                    continue

                if not categories:
                    for product in load_json(os.path.join(root, file)):
                        yield product
                else:
                    for product in load_json(os.path.join(root, file)):
                        if str(product.get("category_id", "")) in categories:
                            yield product

    @classmethod
    def categories_to_products(cls, path: Optional[str] = None, categories: List[str] = []):
        categories_to_products = defaultdict(set)

        for root, dirs, files in os.walk(path):
            for file in files:
                if not file.endswith(".txt"):
                    continue

                if not categories:
                    for product in load_json(os.path.join(root, file)):
                        categories_to_products[str(product["category_id"])].add(product["id"])
                else:
                    for product in load_json(os.path.join(root, file)):
                        if str(product["category_id"]) in categories:
                            categories_to_products[str(product["category_id"])].add(product["id"])

        return categories_to_products

    @classmethod
    def categories_to_products_without_status(cls, keep_status: List[str], path: str, categories: List[str] = []):
        categories_to_products = defaultdict(set)

        for root, dirs, files in os.walk(path):
            for file in files:
                if not file.endswith(".txt"):
                    continue

                for product in load_json(os.path.join(root, file)):
                    if str(product['status']['id']) in keep_status:
                        continue
                    else:
                        # create the dict only for allowed categories or for all of them if there are no allowed categories specified
                        if not categories or str(product["category_id"]) in categories:
                            categories_to_products[str(product["category_id"])].add(product["id"])

        return categories_to_products

    @classmethod
    def get_products_for_offers(cls, offers_path, offer_ids: List[int]):
        product_to_offers = defaultdict(list)
        for root, dirs, files in os.walk(offers_path):
            for file in files:
                if not file.endswith(".txt"):
                    continue

                for offer in load_json(os.path.join(root, file)):
                    if offer["id"] in offer_ids:
                        product_id = int(file.split(".")[1])
                        product_to_offers[product_id].append(offer["id"])
        return product_to_offers

    @classmethod
    def index_products(cls, path: Optional[str] = None, max_products: Optional[int] = -1):
        if not os.path.exists(path):
            raise KeyError(f"{path} does not exist")
        index = 0
        if max_products == -1:
            for root, dirs, files in os.walk(path):
                for file in files:
                    if not file.endswith(".txt"):
                        continue

                    for product in load_json(os.path.join(root, file)):
                        index += 1
                        yield (index, product)
        else:
            for root, dirs, files in os.walk(path):
                for file in files:
                    if not file.endswith(".txt"):
                        continue

                    for product in load_json(os.path.join(root, file)):
                        index += 1
                        if index > max_products:
                            break
                        yield (index, product)
                    else:
                        continue
                    break
                else:
                    continue
                break

    @classmethod
    def offers(cls, path, product_id: Union[int, str]):
        path = f"{path}/product.{product_id}.txt"

        if not os.path.isfile(path):
            return []

        return [o for o in load_json(path)]

    @classmethod
    def all_offers(cls, path):
        for file in os.listdir(path):
            if file.endswith(".txt"):
                file_path = os.path.join(path, file)

                yield [o for o in load_json(file_path)]

    @classmethod
    def offers_by_id(cls, path, selected_ids):
        str_sel_ids = [str(sid) for sid in selected_ids]
        for file in os.listdir(path):
            if file.endswith(".txt"):
                file_path = os.path.join(path, file)

                yield [o for o in load_json(file_path) if str(o["id"]) in str_sel_ids]
            else:
                yield []

    @classmethod
    def n_offers(cls, path, product_id: int):
        path = f"{path}/product.{product_id}.txt"

        if not os.path.isfile(path):
            return 0

        return count_lines(path)

    @classmethod
    def delete_products(cls, offers_path: str, products_path: str, product_ids: List[int]):
        products_file = os.path.join(products_path, "products.txt")
        remove_selected_lines(products_file, "id", product_ids)

        # delete offers files for removed products
        for pr_id in product_ids:
            path = os.path.join(offers_path, f"product.{pr_id}.txt")
            # remove if the file exists
            if os.path.exists(path):
                os.remove(path)

    @classmethod
    def delete_offers(
        cls, offers_path: str, products_path: str, product_id: int, offer_ids: List[int] = [], del_no_offer_product: bool = False
    ):
        path = f"{offers_path}/product.{product_id}.txt"

        if offer_ids:
            n_remaining_lines = remove_selected_lines(path, "id", offer_ids)
        else:
            logging.info(f"Deleting offers file for product {product_id}")
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            n_remaining_lines = 0

        if n_remaining_lines == 0 and del_no_offer_product:
            products_file = f"{products_path}/products.txt"
            n_remaining_lines = remove_selected_lines(products_file, "id", [product_id])
            logging.info(f"Removed product {product_id} after removing offers {offer_ids}")

    @classmethod
    def random_offer(cls, path):
        if cls.offer_files is None:
            cls.offer_files = [
                f"{path}/{f}" for f in listdir(path) if isfile(join(path, f))
            ]
        path = random.choice(cls.offer_files)

        offers = [offer for offer in load_json(path)]

        if len(offers) == 0:
            return None

        return random.choice(offers)


class Corpus:
    files = {}

    @staticmethod
    def n_lines(path):
        return sum(1 for line in open(path))

    @classmethod
    def close(cls):
        for f in cls.files.values():
            f.close()

    @classmethod
    def load(cls, path):
        for line in load_line(path):
            yield line.rstrip()

    @classmethod
    def save(cls, path, line, open_mode="w"):
        if path not in cls.files:
            safe_directory(path)
            cls.files[path] = open(path, open_mode)

        cls.files[path].write(f"{line}\n")

    @classmethod
    def write(cls, path, content):
        safe_directory(path)
        with open(path, "w") as f:
            f.write(content)


def merge_collector_folders(
    directories: Optional[list],
    data_directory: Union[str, Path]
) -> Optional[list]:
    """
    We expect a rigid structure of the collector folder and its' two subfolders:
     - xxx/products containing products.txt
     - xxx/offers containing product.xy.txt files
    All directories of the supplied `directories` is expected to follow this structure.

    If there are multiple directories supplied, they are merged into one named "collector_merged".
    """
    if directories:
        if len(directories) == 1:
            return directories[0]

        final_directory = f"{data_directory}/collector_merged"
        safe_directory(os.path.join(final_directory, "products"))
        safe_directory(os.path.join(final_directory, "offers"))

        with open(os.path.join(final_directory, "products", "products.txt"), 'wb') as fw:
            for directory in directories:
                dir_products_path = os.path.join(directory, "products")
                dir_offers_path = os.path.join(directory, "offers")
                # safety checks
                if not os.path.isdir(dir_products_path) or not os.path.isfile(os.path.join(dir_products_path, "products.txt")):
                    logging.warning(f"Products directory or file is missing in {directory}, skipping merging this category")
                    continue
                if not os.path.isdir(dir_offers_path) or len(os.listdir(dir_offers_path)) == 0:
                    logging.warning(f"Offers directory is missing or is empty in {directory}, skipping merging this category")
                    continue

                with open(os.path.join(directory, "products", "products.txt"), 'rb') as fr:
                    shutil.copyfileobj(fr, fw)

                for offers_file in listdir(os.path.join(directory, "offers")):
                    of = os.path.join("offers", offers_file)
                    shutil.move(
                        os.path.join(directory, of),
                        os.path.join(final_directory, of)
                    )

                shutil.rmtree(directory)

        return final_directory
