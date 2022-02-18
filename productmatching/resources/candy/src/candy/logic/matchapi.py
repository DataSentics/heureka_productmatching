import asyncio
import logging
import json
import tenacity
from time import sleep
from typing import Union, Set, List
from llconfig import Config

from buttstrap.remote_services import RESTService
from buttstrap.remote_services import RemoteServices

from matching_common.clients.base_client import BaseClient
from candy.utils import download_from_s3


class MatchApiManager:
    def __init__(self, config):
        self.matchapi_base = config['MATCHAPI_BASE']
        self.matchapi_conf_address = config['MATCHAPI_CONFIG_FILE']
        self.matchapi_default_unknown = config['MATCHAPI_DEFAULT_UNKNOWN']
        self.mapi_id_to_categories = {}
        self.category_to_mapi_id = {}
        self.mapi_id_to_model_info = {}
        self.matchapi_services = None

    async def discover(self):
        while True:
            matchapi_conf_path = download_from_s3(self.matchapi_conf_address, '/data')
            with open(matchapi_conf_path, 'r') as f:
                matchapi_conf = json.load(f)

            disabled_ids = matchapi_conf.get("DISABLED")
            disabled_ids = disabled_ids.split(",") if disabled_ids else []
            mapi_ids_full = set(matchapi_conf.keys()) - {"DISABLED"}
            mapi_ids = mapi_ids_full - set(disabled_ids)
            logging.info(f"MATCHAPI IDS IN CONFIG {mapi_ids_full}")
            logging.info(f"USED MATCHAPI IDS {mapi_ids}")
            logging.info(f"DISABLED MATCHAPI IDS {disabled_ids}")

            if mapi_ids:
                # trying to locate all specified matchapis in the environment
                await self.close()
                self.matchapi_services = RemoteServices(
                    self.config(mapi_ids),
                    rest=list(mapi_ids),
                )
                await asyncio.gather(self.matchapi_services.init())

                # firstly create the new mappings, assign them when finished, otherwise can lead to temporarily missing default unknown decisions with present model info
                new_mapi_id_to_categories = {}
                new_mapi_id_to_model_info = {}
                for mapi_id in mapi_ids:
                    try:
                        async with self.matchapi_services.get(mapi_id).context as api:
                            categories = await MatchApiClient(
                                matchapi_service=api
                            ).get_categories()
                            model_info = await MatchApiClient(
                                matchapi_service=api
                            ).get_model_info()
                        if categories == matchapi_conf[mapi_id]:
                            new_mapi_id_to_categories[mapi_id] = categories
                            new_mapi_id_to_model_info[mapi_id] = model_info
                        else:
                            raise Exception(
                                f"Categories {categories} for matchapi {mapi_id} differs from categories in cofig {matchapi_conf[mapi_id]}"
                            )
                    except Exception as e:
                        logging.exception(f"Error while retrieving categories from matchapi {mapi_id}, won't be used. {e}")

                # assign new mapping
                self.mapi_id_to_categories = new_mapi_id_to_categories
                self.mapi_id_to_model_info = new_mapi_id_to_model_info

                if not self.mapi_id_to_categories:
                    if self.matchapi_default_unknown:
                        # TODO: change to base mdoel
                        logging.warning("No undisabled MatchAPIs available, redirecting all offers to unknown.")
                        self.matchapi_services = None
                        break
                    else:
                        logging.warning("No undisabled MatchAPIs available, sleeping 5s.")
                        sleep(5)
                        continue

                self.category_to_mapi_id = {
                    category: mapi_id
                    for mapi_id, categories in self.mapi_id_to_categories.items()
                    for category in categories.strip().split(',')
                }

                logging.info("SUCCESFULLY RETRIEVED CATEGORIES FROM MATCHAPIS")
                logging.info(f"MatchAPI id to categories: {str(self.mapi_id_to_categories)}")
                logging.info(f"Category to MatchAPI id: {str(self.category_to_mapi_id)}")
                break
            else:
                if self.matchapi_default_unknown:
                    # TODO: change to base model
                    logging.warning("No functional MatchAPIs available, redirecting all offers to unknown.")
                    break
                else:
                    logging.warning("No functional MatchAPIs available, sleeping 5s.")
                    sleep(5)

    async def discover_loop(self):
        while True:
            await asyncio.sleep(10 * 60)
            _ = await asyncio.wait_for(self.discover(), timeout=None)

    def get_categories(self) -> list:
        return list(self.category_to_mapi_id.keys())

    def get_matchapi_ids_from_categories(self, categories) -> list:
        # ["0"] if there are no MatchaAPIs available, leads to "DEFAULT" model
        def_id = self.category_to_mapi_id.get("DEFAULT")
        return list({self.category_to_mapi_id.get(c, def_id) for c in categories} - {None})

    def config(self, mapi_ids) -> Config:
        conf = Config()
        for i in mapi_ids:
            conf.init(i, str, self.matchapi_base.replace('@@', i))

        return conf

    def get_contexts(self, categories: Union[Set[str], List[str]]) -> dict:
        # return the passed cats as "1,2,3" and corresponding contexts, one context possibly for multiple categories
        contexts = {
            self.mapi_id_to_categories[mapi_id]: self.get_context_from_id(mapi_id)
            for mapi_id in self.get_matchapi_ids_from_categories(categories)
        }
        return contexts

    def get_context_from_id(self, mapi_id):
        return self.matchapi_services.get(mapi_id).context if self.matchapi_services else None

    def get_context_from_category(self, category: str):
        mapi_id = self.category_to_mapi_id[category]
        return self.get_context_from_id(mapi_id)

    def get_model_info_from_id(self, mapi_id: str):
        model_info = self.mapi_id_to_model_info.get(mapi_id)
        return model_info

    async def close(self):
        if self.matchapi_services:
            await self.matchapi_services.close_all()


class MatchApiClient(BaseClient):
    MATCH_ENDPOINT = "match"
    MATCH_MANY_ENDPOINT = "match-many"
    CATEGORIES_ENDPOINT = "categories"
    MODEL_INFO_ENDPOINT = "model-info"

    def __init__(self, matchapi_service: RESTService):
        self.service = matchapi_service
        super().__init__(service_name="matchapi")

    async def get_match(self, offer: dict, product: dict, explain: bool = True) -> dict:
        params = {
            "explain": explain,
            "offer": {
                "name": offer["match_name"],
                "price": offer["price"],
                "ean": offer["ean"],
                "shop": offer["shop_id"],
                "attributes": {a["name"]: str(a["value"]) for a in offer.get("attributes", [])},
                "parsed_attributes": {a["name"]: str(a["value"]) for a in offer.get("parsed_attributes", [])},
                "image_url": offer.get("image_url", ""),
                "external_image_url": offer.get("external_image_url",""),
            },
            "product": {
                "name": product["data"]["name"],
                "category_id": str(product["data"]["category_id"]),
                "prices": product["data"]["prices"] or [],
                "eans": product["data"]["eans"] or [],
                "shops": product["data"]["shops"] or [],
                "attributes": {a["name"]: str(a["value"]) for a in product["data"]["attributes"] or []},
                "ean_required": product["ean_required"],
                "unique_names": product["unique_names"],
                "image_url": product.get("image_url", ""),
            },
        }

        response = await self.service.request(
            "POST",
            self.MATCH_ENDPOINT,
            headers=self.headers,
            json=params,
        )

        response = await self._process_response(response)

        return response

    @tenacity.retry(
        reraise=True,
        stop=tenacity.stop_after_attempt(5),
        wait=tenacity.wait_random(min=0, max=1))
    async def get_categories(self):

        response = await self.service.request(
            "GET",
            self.CATEGORIES_ENDPOINT,
            headers=self.headers,
            json={"explain": True},
        )

        response = await self._process_response(response)

        return response

    @tenacity.retry(
        reraise=True,
        stop=tenacity.stop_after_attempt(5),
        wait=tenacity.wait_random(min=0, max=1))
    async def get_model_info(self):

        response = await self.service.request(
            "GET",
            self.MODEL_INFO_ENDPOINT,
            headers=self.headers,
            json={"explain": True},
        )

        response = await self._process_response(response)

        return response

    async def get_match_many(self, offer: dict, products: list, explain: bool = True) -> dict:
        params = {
            "explain": explain,
            "offer": {
                "name": offer["match_name"],
                "price": offer["price"],
                "ean": offer["ean"],
                "shop": offer["shop_id"],
                "attributes": {a["name"]: str(a["value"]) for a in offer["attributes"] or []},
                "image_url": offer.get("image_url", ""),
                "external_image_url": offer.get("external_image_url",""),
            },
            "products": [{
                "name": product["data"]["name"],
                "category_id": str(product["data"]["category_id"]),
                "prices": product["data"]["prices"] or [],
                "eans": product["data"]["eans"] or [],
                "shops": product["data"]["shops"] or [],
                "attributes": {a["name"]: str(a["value"]) for a in product["data"]["attributes"] or []},
                "ean_required": product.get("ean_required", False),
                "unique_names": product.get("unique_names", False),
                "image_url": product.get("image_url", ""),
            } for product in products],
        }

        # this takes very long, only possible improvement lies in matching model modifications
        response = await self.service.request(
            "POST",
            self.MATCH_MANY_ENDPOINT,
            headers=self.headers,
            json=params,
        )

        response = await self._process_response(response)

        return response
