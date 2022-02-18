import os
import sys
import fastapi
import json
import logging
import pydantic
import uvicorn
import typing as t
import asyncio
import time

from prometheus_fastapi_instrumentator import Instrumentator

from utilities.args import str2bool

FEATURE_OC = os.getenv('MATCHAPI_FEATURE_OC', False)

logging.basicConfig(level=logging.INFO)


def create_matchapi_app() -> fastapi.applications.FastAPI:
    """
    Created FastApi app with matchapi from either model either registered in mlflow or from local artifacts and code.
    Used when deploying matchapi and also in tests. TransformerApi is not implemented for now because of problems with async pyfunc model retrieve.
    """
    app = fastapi.FastAPI()

    if str2bool(os.getenv("CONFIG__DEBUG", True)):
        # loading model from local paths, mostly for local debuging
        # to run this in kube, models have to be downloaded e.g. in pod initialization
        # and paths to models have to be changed accordingly
        from preprocessing.models import exact_match
        from utilities.attributes import Attributes
        from xgboostmatching.models.model import XGBoostMatchingModel

        logging.info("Using models from specified files.")
        # for local testing, you don't want to download sbert very often
        namesimilarity = exact_match.ExactMatchModel(
            tok_norm_args=f'input_pmi={os.getenv("CONFIG__pmi", "/app/data/pmi.txt")}'
        )
        attributes = Attributes(
            from_=os.getenv("CONFIG__attributes", "/app/data/attributes.json"),
        )
        model = XGBoostMatchingModel(
            xgboost_path=os.getenv("CONFIG__xgboost", "/app/data/xgboostmatching_model/best.xgb"),
            namesimilarity=namesimilarity,
            attributes=attributes,
            feature_oc=FEATURE_OC,
            tok_norm_args=f'input_pmi={os.getenv("CONFIG__pmi", "/app/data/pmi.txt")}',
            unit_conversions=str2bool(os.getenv("CONFIG__unit_conversions", True)),
            price_reject_a=float(os.getenv("CONFIG__price_reject_a", '1000.0')),
            price_reject_b=float(os.getenv("CONFIG__price_reject_b", '400.0')),
            price_reject_c=float(os.getenv("CONFIG__price_reject_c", '2.5'))
        )
        matchapi_categories = "9999"  # fill as required for your testing
        model_info_message = "No model version info."
    else:
        logging.info("Using model from MLFlow model registry.")
        model_name = "matching_model"
        # TODO: the model used in actrual production will probably be a different one
        model_stage = 'Production'
        matchapi_id = os.getenv("MATCHAPI_ID").strip()
        with open("/app/data/matchapi_id_categories_mapping.json", "r") as f:
            categories_config = json.load(f)

        logging.info(f"Current id-categories mapping: {categories_config}")
        matchapi_categories = categories_config[matchapi_id]
        logging.info(f"Categories served by this MatchAPI: {matchapi_categories}")
        tags = {"categories": matchapi_categories}

        from utilities.model_registry.client import MlflowRegistryClient
        registry_client = MlflowRegistryClient()

        # modify the current sys path, this prevents any possible compatibility issues
        SP_paths = [p for p in sys.path if "SourcesPython" in p]
        for p in SP_paths:
            sys.path.remove(p)

        # delete imported modules, ensure use of RegistryMatchingModelConstructor defined in registered code
        modules_to_del = [m for m in sys.modules if 'preprocessing' in m or 'xgboostmatching' in m or 'utilities' in m]
        for md in modules_to_del:
            del sys.modules[md]

        model_info = registry_client.get_model_info_stage(model_name, model_stage, tags)
        model_info_message = {"name": model_info.name, "version": model_info.version, "tags": model_info.tags}
        logging.info(f"model info: {model_info}")
        model = registry_client.get_model_by_stage(model_name, model_stage, tags)

    # import from registered code
    from xgboostmatching.models.features import Product, Offer
    from xgboostmatching.models.model import Match

    class MatchRequest(pydantic.BaseModel):
        product: Product
        offer: Offer

    class MatchManyRequest(pydantic.BaseModel):
        products: t.List[Product]
        offer: Offer

    product = Product.parse_obj({'name': "Bufo Bufo", "category_id": "1235"})
    offer = Offer.parse_obj({"name": "Ufo Bufo", "price": 66.6, "shop": "endless pain"})
    logging.info(f'TESTING PREDICTION {model.predict({"product": product, "offer": offer})}')

    # expose app
    Instrumentator().instrument(app).expose(app)

    @app.get("/test")
    async def test():
        def fn():
            while 1:
                st = time.time()
                logging.info(f"test {time.time()-st}")
                time.sleep(0.5)
        timeout = 3
        loop = asyncio.get_event_loop()
        return await asyncio.wait_for(loop.run_in_executor(None, fn), timeout=timeout)

    # define endopints
    @app.get("/ping")
    def ping() -> str:
        return "pong"

    @app.get("/v1/categories")
    def categories() -> str:
        return matchapi_categories

    @app.get("/v1/model-info")
    def model_info() -> str:
        return model_info_message

    @app.post("/v1/match")
    def match(request: MatchRequest) -> Match:
        return model.predict({"product": request.product, "offer": request.offer})

    @app.post("/v1/match-many")
    def match_many(request: MatchManyRequest) -> t.List[Match]:
        res = [model.predict({"product": product, "offer": request.offer}) for product in request.products]
        return res

    return app


if __name__ == "__main__":

    # create app with matchapi
    app = create_matchapi_app()
    # start uvicorn, start here instead of docker compose run to prevent problems with asyncio loops
    uvicorn.run(app, host="0.0.0.0", port=8080)
