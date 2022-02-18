import faiss
import time
import numpy as np
import logging
from collections import defaultdict, namedtuple
from time import time
from typing import List, Union

from utilities.preprocessing import Pipeline
from utilities.loader import Product
from preprocessing.models.transformer import TransformerModel
from utilities.cs2_downloader import CS2Downloader

from matching_common.clients.candidate import Candidate
from matching_common.clients.provider import FAISS_CANDIDATES_SOURCE
from buttstrap.remote_services import RemoteServices


Search = namedtuple("Search", "faiss transformer i_to_id i_to_name id_to_name")


def determine_build_index(offers_path):
    for one_file_offers in Product.all_offers(offers_path):
        for offer in one_file_offers:
            if FAISS_CANDIDATES_SOURCE not in offer.get("candidates_sources", {}):
                return True
    return False


class SearchDataDownloader(CS2Downloader):
    def __init__(
        self,
        categories: list,
        remote_services: RemoteServices,
    ):
        self.categories = categories
        super().__init__(remote_services)

    async def product_search_download(self, limit=200, max_products_per_category=np.inf):
        for category_id in self.categories:
            logging.info(f"starting download of category {category_id}")
            fields = ["id", "name"]
            status = [11, 12, 13, 14, 15, 16]

            async for products in self.products_download_range(
                category_id, status=status, fields=fields, max_products=max_products_per_category, limit=limit
            ):
                yield products


class FaissClient():
    def __init__(
        self,
        tok_norm_args: str,
        input_transformer: Union[str, TransformerModel],
        downloader_params: dict = {},
        available_data_path: str = None,
        build_index: bool = True
    ):
        assert available_data_path or downloader_params
        self.tok_norm_args = tok_norm_args
        self.downloader_params = downloader_params
        self.candidates_source = "faiss"
        self.available_data_path = available_data_path
        self.build_index = build_index

        if not isinstance(input_transformer, TransformerModel):
            self.transformer_model = TransformerModel(input_transformer)
        else:
            self.transformer_model = input_transformer

    async def close(self):
        await self.transformer_model.close()

    async def init(self):
        def process_item(item: dict, vectorize: bool = True):
            nonlocal index_in
            name = self.preprocessing(item["name"])
            i_to_id[index_in] = item["id"]
            i_to_name[index_in] = name
            index_in += 1
            if vectorize:
                to_vectorize.append(name)

        self.preprocessing = Pipeline.create(self.tok_norm_args)
        i_to_id = dict()
        i_to_name = dict()
        id_to_name = dict()
        to_vectorize = []
        vectors = []
        index_in = 0
        vectorize = True

        if not self.build_index:
            logging.info("FAISS index not built. Candidates will be extracted directly and solely from offers data.")
            vectorize = False

        if self.available_data_path:
            for item in Product.products(self.available_data_path):
                process_item(item, vectorize)
        else:
            self.downloader = SearchDataDownloader(**self.downloader_params)
            async for items in self.downloader.product_search_download():
                for item in items:
                    process_item(item, vectorize)

        faiss_index = None
        if self.build_index:
            await self.transformer_model.init()
            logging.info(f"Collected {len(to_vectorize)} products to index")
            st = time()
            vectors = await self.transformer_model.get_sentence_vector(to_vectorize)
            logging.info(f"Creating embeddigns took {time()-st}s")

            logging.info(f"Ending faiss indexing, indexed {index_in} products in {time() - st}s")

            faiss_index = faiss.IndexFlatL2(await self.transformer_model.get_dimension())
            faiss_index.add(np.array(vectors, dtype=np.float32))

        id_to_name = {i_to_id[i]: i_to_name[i] for i in i_to_id}
        self.search = Search(faiss_index, None, i_to_id, i_to_name, id_to_name)

    async def get_candidates(self, items: List[dict], limit: int, **kwargs) -> dict:
        def get_id_from_sim_index(sim_index):
            if sim_index != -1:
                return str(self.search.i_to_id[sim_index])

        similiarity_limit = kwargs.get("similiarity_limit")
        similiarity_limit = float(similiarity_limit) if similiarity_limit else -1
        value_field = kwargs.get('value_field', "match_name")

        candidates = defaultdict(dict)

        for item in items:
            name = item[value_field]
            if "faiss" in item.get("candidates_sources", {}):
                # get only faiss candidates, assuming we only want faiss candidates when calling FaissClient
                candidates_ids = item["candidates_sources"]["faiss"]
                distances = [0] * len(candidates_ids)
            else:
                vector = np.array([await self.transformer_model.get_sentence_vector(name)], dtype=np.float32)
                found = self.search.faiss.search(vector, k=limit)
                distances, sim_indexes = found[0][0], found[1][0]
                candidates_ids = [get_id_from_sim_index(sim_index) for sim_index in sim_indexes]

            for distance, candidate_id in zip(distances, candidates_ids):
                if candidate_id and (similiarity_limit <= 0 or distance < similiarity_limit):
                    candidate = Candidate(
                        id=candidate_id,
                        source=[self.candidates_source],
                        distance=distance
                    )
                    candidates[str(item['id'])][candidate.id] = candidate
        return candidates

    async def get_candidates_names(self, items: List[dict], limit: int, **kwargs) -> dict:
        def format_names(candidates_dict):
            res = [self.search.id_to_name.get(int(c_id)) for c_id in candidates_dict]
            return [r for r in res if r]

        candidates = await self.get_candidates(items, limit, **kwargs)
        candidates_names = {
            item_id: format_names(candidates_dict)
            for item_id, candidates_dict in candidates.items()
        }
        return candidates_names
