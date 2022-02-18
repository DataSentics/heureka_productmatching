import logging
import typing as t
import numpy as np

from buttstrap.remote_services import RemoteServices
from matching_common.clients.cs2_client import CatalogueServiceClient

from utilities.helpers import split_into_batches


class CS2Downloader():
    def __init__(self, remote_services: RemoteServices):
        self.remote_services = remote_services
        self.catalogue_client = CatalogueServiceClient

    async def offers_download(self, offer_ids: t.Union[list, set], fields: list = [], status="all", batch_len=20):
        if not fields:
            fields = [
                "id", "name", "match_name", "price",  "url", "product_id",
                "parsed_attributes.id", "parsed_attributes.name", "parsed_attributes.value", "parsed_attributes.unit",
                "attributes.id", "attributes.name", "attributes.value", "attributes.unit",
                "ean", "shop_id", "image_url", "external_image_url", "url"
            ]
        parameters = {"status": status}
        for offer_ids_batch in split_into_batches(offer_ids, batch_len):
            try:
                async with self.remote_services.get('catalogue').context as catalogue:
                    items = await self.catalogue_client(
                        catalogue_service=catalogue
                    ).get_offers(offer_ids_batch, fields, parameters)

                yield items
            except Exception:
                logging.exception("CS2 gone away. Offers data not downloaded.")

    async def products_download(self, item_ids, status="all", fields: list = [], return_params: bool = False, parameters: dict = {}, batch_len=1):
        # using category.slug field is much faster than category_slug due to internal joins in CS2
        if not fields:
            fields = [
                "id", "category_id", "name", "prices", "slug", "category.slug",
                "attributes.id", "attributes.name", "attributes.value", "attributes.unit",
                "eans", "shops", "image_url", "status", "producers", "offers_count"
            ]

        # be aware, that passing empty list to `item_ids` will download a batch of 'random' products with lowest ids
        if item_ids:
            params = [
                {'id': items_batch, 'fields': fields, 'status': status}
                for items_batch in split_into_batches(item_ids, batch_len)
            ]
        else:
            logging.warning("Downloading batch of 'random' products with lowest ids.")
            params = [{'id': [], 'fields': fields, 'status': status}]

        # TODO: handle these tries more nicely
        async with self.remote_services.get('catalogue').context as catalogue:
            for param in params:
                try:
                    product = await self.catalogue_client(
                        catalogue_service=catalogue
                    ).get_products(param["id"], param["fields"], param["status"], parameters=parameters)
                    if return_params:
                        yield product, param
                    else:
                        yield product
                except Exception:
                    logging.exception(f"CS2 gone away. Products data not downloaded. {param['id']}")

    async def products_download_range(
        self, parent_category_id, products_from_id: int = 0, status="all",
        fields: list = ["id", "name"], max_products=np.inf, limit: int = 200
    ):
        n_products = 0
        async with self.remote_services.get('catalogue').context as catalogue:
            while True:
                try:
                    parameters = {
                        "products_from_id": products_from_id,
                        "parent_category_id": parent_category_id,
                        "order_by": "id:asc",
                        "limit": limit,
                        }
                    products = await self.catalogue_client(
                        catalogue_service=catalogue
                    ).get_products([], fields, status, parameters=parameters)
                    if not products:
                        break
                    n_products += len(products)
                    if n_products > max_products:
                        yield products[:(max_products-n_products)]
                        break

                    yield products

                    if n_products == max_products:
                        break

                    products_from_id = products[-1]['id'] + 1

                except Exception:
                    logging.exception("CS2 gone away. Products data not downloaded.")

    async def category_info(self, category_id: str, fields: t.List[str]):
        params = {
            "categories": [str(category_id)],
            "fields": fields,
        }
        try:
            async with self.remote_services.get('catalogue').context as catalogue:
                category_info = await self.catalogue_client(
                    catalogue_service=catalogue
                ).get_categories_info(**params)
                return category_info
        except Exception:
            logging.exception("CS2 gone away. Category info not retrieved.")

    async def products_offers(self, product_id: str):
        try:
            async with self.remote_services.get('catalogue').context as catalogue:
                products_offers = await self.catalogue_client(
                    catalogue_service=catalogue
                ).get_products_offers(product_id)
                return products_offers
        except Exception:
            logging.exception("CS2 gone away. Category info not retrieved.")
