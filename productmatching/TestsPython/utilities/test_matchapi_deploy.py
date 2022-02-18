"""
Tests matchapi loading and its functionality.
Uses mlflow model registry and selects model based on tags and name.
"""
import pytest
import sys
import json
import shutil
from pathlib import Path

sp_path = "/app/SourcesPython"

sys.path.append(sp_path)

# somewhat hacky solution for import related issues
# to prevent import from code that was not registered together with model
for fol in ['preprocessing', 'xgboostmatching']:
    shutil.rmtree(f"{sp_path}/{fol}")

from matchapi.main import create_matchapi_app
from starlette.testclient import TestClient


DATA_DIR = Path(__file__).resolve().parents[1] / "dataset"
MATCHAPI_ITEMS_PATH = DATA_DIR / "utilities" / "matchapi_items.json"

# load items for matchapi testing
@pytest.fixture(scope="module")
def items():
    with open(str(MATCHAPI_ITEMS_PATH), "r") as f:
        items = json.load(f)
    return items


@pytest.fixture(scope="module")
def test_app():
    # initialize the app, load needed models
    app = create_matchapi_app()
    client = TestClient(app)
    yield client


def test_liveness(test_app):
    response = test_app.get("/ping")
    assert response.status_code == 200
    assert response.json() == "pong"


# text /v1/categories endpoint


def test_categories(test_app):
    response = test_app.get("/v1/categories")
    assert response.status_code == 200


# text /v1/model-info endpoint


def test_model_info(test_app):
    response = test_app.get("/v1/model-info")
    assert response.status_code == 200


# test 'v1/match' endpoint


def test_empty(test_app):
    # empty product and offer
    data = {
        "product": {},
        "offer": {}
    }
    response = test_app.post("/v1/match", data=json.dumps(data))
    # expecting Unprocessable Entity
    assert response.status_code == 422


def test_empty_product(test_app, items):
    # empty product, nonempty offer
    data = {
        "product": {},
        "offer": items["offer_same"]
    }
    response = test_app.post("/v1/match", data=json.dumps(data))
    # expecting Unprocessable Entity
    assert response.status_code == 422


def test_empty_offer(test_app, items):
    # empty offer, nonempty product
    data = {
        "product": items["product"],
        "offer": {}
    }
    response = test_app.post("/v1/match", data=json.dumps(data))
    # expecting Unprocessable Entity
    assert response.status_code == 422


def test_match_list_offers(test_app, items):
    # try response when list of offers given
    data = {
        "product": items["product"],
        "offer": [items["offer_same"]]
    }
    response = test_app.post("/v1/match", data=json.dumps(data))
    # expecting Unprocessable Entity
    assert response.status_code == 422


def test_match_list_products(test_app, items):
    # try response when list of products given instead of using match-many endpoint
    data = {
        "product": [items["product"]],
        "offer": items["offer_same"]
    }
    response = test_app.post("/v1/match", data=json.dumps(data))
    # expecting Unprocessable Entity
    assert response.status_code == 422


# def test_match_ean_required(test_app, items):
#     # test response for offer the same as the product
#     data = {
#         "product": items["product_ean_required"],
#         "offer": items["offer_ean"]
#     }
#     response = test_app.post("/v1/match", data=json.dumps(data))
#     response_body = response.json()

#     assert response.status_code == 200
#     assert response_body["match"] == "yes"
#     assert response_body["details"] == "Ean match enabled and offer ean present among product eans."


def test_match_unique_name(test_app, items):
    # test response for offer the same as the product
    data = {
        "product": items["product_unique_names"],
        "offer": items["offer_same"]
    }
    response = test_app.post("/v1/match", data=json.dumps(data))
    response_body = response.json()

    assert response.status_code == 200
    assert response_body["match"] == "yes"
    # ensure compatibility with older model version, where 1.1 was swapped with 1.3
    assert response_body["details"] in ["Name match: namesimilarity=1.1", "Name match: namesimilarity=1.2", "Name match: namesimilarity=1.3"]


def test_match_different(test_app, items):
    # test response for offer completely different from the product
    data = {
        "product": items["product"],
        "offer": items["offer_different"]
    }
    response = test_app.post("/v1/match", data=json.dumps(data))
    response_body = response.json()
    assert response.status_code == 200
    # expected it was not paired
    assert response_body["match"] != "yes"


def test_match_unit_mismatch(test_app, items):
    # test response for offer completely different from the product
    data = {
        "product": items["product"],
        "offer": items["offer_unit_mismatch"]
    }
    response = test_app.post("/v1/match", data=json.dumps(data))
    response_body = response.json()
    assert response.status_code == 200
    # can be unknown or no depending on the model, but it should not be yes
    assert response_body["match"] != "yes"


def test_match_using_xgboost(test_app, items):
    # test response for similar offer, so xgboost model will be called,
    # it depends on the namesimilarity model, whether the offer will pass the hard checks
    # mainly to check to update in s3 url addresses for OC, since we use mlflow model registry
    data = {
        "product": items["product"],
        "offer": items["offer_similar"]
    }
    response = test_app.post("/v1/match", data=json.dumps(data))
    # check only if it passed the model
    assert response.status_code == 200


# test 'v1/match-many' endpoint

def test_many_empty(test_app):
    # empty product and offer
    data = {
        "products": {},
        "offer": {}
    }
    response = test_app.post("/v1/match-many", data=json.dumps(data))
    # expecting Unprocessable Entity
    assert response.status_code == 422


def test_match_many_different(test_app, items):
    # test response for offer the same as the product
    data = {
        "products": [items["product"], items["product_different"]],
        "offer": items["offer_same"]
    }
    response = test_app.post("/v1/match-many", data=json.dumps(data))
    response_body = response.json()

    assert response.status_code == 200
    # we should not get the same results
    assert [r["match"] for r in response_body] in [["no", "no"], ["unknown", "no"], ["yes", "no"], ["no", "unknown"], ["unknown", "unknown"], ["yes", "unknown"]]


def test_match_many_list_offers(test_app, items):
    # test response when trying to pass more offers
    data = {
        "products": [items["product"]],
        "offer": [items["offer_same"], items["offer_same"]]
    }
    response = test_app.post("/v1/match-many", data=json.dumps(data))
    # expecting Unprocessable Entity
    assert response.status_code == 422
