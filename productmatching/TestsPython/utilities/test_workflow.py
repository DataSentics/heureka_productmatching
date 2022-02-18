"""
Tests all components of training workflow.
At the end, it creates matchapi with trained models and tests it.
"""
import os
import shutil
import pytest
import sys
import random
import json
import ujson
from copy import deepcopy
from argparse import Namespace
from pathlib import Path

sys.path.append("/app/SourcesPython")
sys.path.append("/app/resources/candy/src")

from utilities.loader import read_lines_with_int, write_lines

DATA_DIR = Path(__file__).resolve().parents[1] / "dataset"

COLLECTOR_DATA_PATH = DATA_DIR / "workflow_data" / "collector"
MATCHAPI_ITEMS_PATH = DATA_DIR / "utilities" / "matchapi_items.json"


# load items for matchapi testing
@pytest.fixture(scope="module")
def items():
    with open(str(MATCHAPI_ITEMS_PATH), "r") as f:
        items = json.load(f)
    return items


# arguments to all function below, model training is very short and n products is lower
@pytest.fixture(scope="module")
def args(tmp_path_factory):

    base_dir = tmp_path_factory.getbasetemp()
    params = "booster=gbtree,objective=binary:logistic,eta=0.3,max_depth=6,subsample=0.5,seed=0"
    args = Namespace(
        base_dir=base_dir,
        input_collector=str(COLLECTOR_DATA_PATH),
        data_directory=str(tmp_path_factory.getbasetemp()),
        preceding_corpus=None,
        # dataset split
        preceding_test_items=None,
        max_test_items_in_category=10,
        # pmi
        input_embedding_dataset=str(base_dir / "embedding_dataset"),
        min_token_frequency=22,
        min_token_length=4,
        min_pmi_score=10,
        # attributes
        preceding_attributes=None,
        # fasttext_train, model used for evaluation
        model="skipgram",
        dim=10,  # lower
        lr=0.05,
        ws=5,
        epoch=1,  # lower
        min_count=1,
        minn=2,
        maxn=3,
        neg=5,
        # candidates retrieval
        ml_paired_offers_to_remove={},
        # dataset split
        train_size=0.8,
        # xgboostmatching_dataset
        # args input_collector_products nd input_collector_offers are constructed near function start
        # because of changes in input_collector_data during tests
        input_fasttext=str(base_dir / "fasttext.bin"),
        input_attributes=str(base_dir / "attributes.json"),
        tok_norm_args=f'input_pmi={str(base_dir / "pmi.txt")}',
        max_products=10,
        candidates_sources=["elastic"],
        job_spec="0/1",
        coros_batch_size=5,
        products_frac=1.0,
        min_product_offers=1,
        max_sample_offers_per_product=10,
        # xgboostmatcjhing update
        extra_features={"match_attributes": [0, 1], "get_namesimilarity_result": [0]},
        input_datasets=[base_dir / "xgboostmatching_dataset_0"],  # for xgb train as well
        preceding_input_collector_products=None,
        preceding_input_collector_offers=None,
        # xgboostmatching_train
        input_datasets_extra=[base_dir / "xgboostmatching_dataset_extra_0"],
        parameters=params,
        iterations=2,
        randomized_search_iter=-1,
        n_components="mle",
        preceding_input_datasets=None,
        # evaluation
        n_offers_to_pair=10,
        category_ids=os.environ.get("CATEGORIES"),
        input_collector_products=str(COLLECTOR_DATA_PATH),
        input_collector_offers=str(COLLECTOR_DATA_PATH),
        use_collector_data=True,
        input_xgb=(base_dir / "xgboostmatching_model" / "best.xgb"),
        similarity_limit=5,
        max_candidates=10,
        catalogue_service_2="http://catalogue-catalogue-service2.cz.k8s.heu.cz",
        finetune_thresholds=False,
        thresholds_path=None,
        prioritize_status=True,
        prioritize_name_match=True,
        remove_longtail_candidates=True,
        test_items_ids_file=str(base_dir / "test_items.list"),
        test_items_data_file=str(base_dir / "test_items_data" / "test_items_data.txt"),
        preceding_test_items_ids_file=None,
        preceding_test_items_data_file=None,
        unit_conversions=True,
        matched_confidence_threshold=.75,
        precision_confidence_threshold=.95,
        per_category_results_to_compare=None,
        # comparison
        excel_path=str(base_dir / "matching_results.xlsx"),
        validation_excel_path=str(base_dir / "matching_results.xlsx"),
        output_filename="comparison_excel.xlsx",
        # model_registration
        categories=os.environ.get("CATEGORIES"),
    )
    return args


@pytest.fixture(scope="module")
def test_app(args):
    # set variables with files paths
    os.environ["CONFIG__DEBUG"] = "true"
    os.environ["CONFIG__xgboost"] = str(args.input_xgb)
    os.environ["CONFIG__attributes"] = str(args.input_attributes)
    os.environ["CONFIG__tok_norm_args"] = str(args.tok_norm_args)
    os.environ["CONFIG__unit_conversions"] = "true"
    os.environ["CONFIG__price_reject_a"] = "1000.0"
    os.environ["CONFIG__price_reject_b"] = "400.0"
    os.environ["CONFIG__price_reject_c"] = "2.5"

    from SourcesPython.matchapi.main import create_matchapi_app
    from starlette.testclient import TestClient
    # initialize the app, load needed models
    app = create_matchapi_app()
    client = TestClient(app)
    yield client


def test_preprocessing_embedding_dataset(args):
    from preprocessing.components.embedding_dataset.main import save_titles

    save_titles(args)


def test_preprocessing_pmi_dataset(args):
    from preprocessing.components.pmi.main import pmi

    pmi(args)


def test_preprocessing_fasttext_train(args):
    from preprocessing.components.fasttext_train.main import train_fasttext

    train_fasttext(args)


def test_candidates_retrieval(args):
    from candidates_retrieval.main import candidates_retrieval

    collector_candidates_data_path = os.path.join(
        os.path.dirname(args.input_collector),
        "collector_candidates"
    )
    # remove previously existing dir e.g. during local testing
    if os.path.isdir(collector_candidates_data_path):
        shutil.rmtree(collector_candidates_data_path)

    candidates_retrieval(args)

    # update collector data path
    args.input_collector = collector_candidates_data_path


# function from workflow, run one by one in order as in workflow (manually ordered)
def test_dataset_split(args):
    from dataset_split.main import separate_test_items

    train_collector_data_path = os.path.join(
        os.path.dirname(args.input_collector),
        "train_collector_data"
    )
    # remove previously existing dir e.g. during local testing
    if os.path.isdir(train_collector_data_path):
        shutil.rmtree(train_collector_data_path)

    separate_test_items(args)

    # update collector data path
    args.input_collector = train_collector_data_path


def test_dataset_split_retrain(args):
    preceding_test_items = read_lines_with_int(args.test_items_ids_file)
    # add previously created test items as preceding data
    args.preceding_test_items_ids_file = args.test_items_ids_file

    from dataset_split.main import separate_test_items

    # copy arguments and put all offers to train set to test including only preceding test items
    temp_args = deepcopy(args)
    temp_args.train_size = 1

    separate_test_items(temp_args)

    # load new results
    new_test_items = read_lines_with_int(args.test_items_ids_file)
    assert new_test_items == preceding_test_items, "Dataset split added new test items when run with same data, check handling preceding_test_items"


def test_preprocessing_extract_attributes(args):
    from preprocessing.components.extract_attributes.main import extract

    extract(args)


def test_preprocessing_extract_attributes_retrain(args):
    # retrain mode, use just created attributes as preceding, nothing new should be added
    args.preceding_attributes = args.input_attributes
    # check whether nothing new was added: load preceding file
    with open(args.input_attributes, "r") as attrfile:
        category_to_name_preceding = ujson.load(attrfile)

    from preprocessing.components.extract_attributes.main import extract

    extract(args)
    # load new file, check equality
    with open(args.input_attributes, "r") as attrfile:
        category_to_name_new = ujson.load(attrfile)

    assert category_to_name_preceding == category_to_name_new, "Extract attributes added new attributes to preceding output with same data, check handling preceding_attributes"


def test_xgboostmatching_dataset(args):
    args.data_directory = Path(args.data_directory)
    args.input_collector = Path(args.input_collector)
    args.input_collector_products = args.input_collector / "products"

    # max products (only in thi step) to make it shorter
    args.max_products = 4

    from xgboostmatching.components.dataset.main import main
    main(args)


def test_xgboostmatching_dataset_update(args):
    args.data_directory = Path(args.data_directory)
    args.input_collector = Path(args.input_collector)
    args.input_collector_products = args.input_collector / "products"
    args.input_collector_offers = args.input_collector / "offers"

    from xgboostmatching.components.dataset.update import main
    main(args)


def test_xgboostmatching_train(args):
    args.data_directory = Path(args.data_directory)
    from xgboostmatching.components.train.main import xgboostmatching_train

    xgboostmatching_train(args)


def test_xgboostmatching_train_retrain(args):
    args.data_directory = Path(args.data_directory)
    # add previously created data as preceding data
    args.preceding_input_datasets = args.input_datasets
    # remove previously trained model to avoid mixing it with model saved during training in callbacks
    xgbmodel_dir = args.data_directory / "xgboostmatching_model"
    if os.path.isdir(xgbmodel_dir):
        shutil.rmtree(xgbmodel_dir)

    from xgboostmatching.components.train.main import xgboostmatching_train

    xgboostmatching_train(args)


def test_evaluation(args):
    args.input_collector_products = args.input_collector / "products"
    args.input_collector_offers = args.input_collector / "offers"

    # select only few items for evaluation
    random.seed(10)
    test_items = read_lines_with_int(args.test_items_ids_file)
    test_items_flt = random.sample(test_items, k=min(args.n_offers_to_pair, len(test_items)))
    write_lines(args.test_items_ids_file, test_items_flt)

    # import and run evaluation
    from evaluation.model_evaluation.main import main
    args.data_directory = str(args.data_directory)
    args.price_reject_a = 1000.0
    args.price_reject_b = 400.0
    args.price_reject_c = 2.5
    main(args)


def test_evaluation_retrain(args):
    args.per_category_results_to_compare = str(args.base_dir / "overall_per_category_results.csv")
    # use the same items both as preceding and new
    args.preceding_test_items_ids_file = args.test_items_ids_file
    args.preceding_test_items_data_file = args.test_items_data_file

    # import and run evaluation
    from evaluation.model_evaluation.main import main

    args.data_directory = str(args.data_directory)
    main(args)


def test_comparison(args):
    from comparison.main import main
    main(args)


def test_liveness(test_app):
    response = test_app.get("/ping")
    assert response.status_code == 200
    assert response.json() == "pong"


# test /v1/categories endpoint


def test_categories(test_app):
    response = test_app.get("/v1/categories")
    assert response.status_code == 200
    # constructed manually, category filled manually
    assert response.json() == "9999"


# test /v1/model-info endpoint


def test_model_info(test_app):
    response = test_app.get("/v1/model-info")
    assert response.status_code == 200
    # constructed manually, no model info included
    assert response.json() == "No model version info."


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


def test_match_ean_required(test_app, items):
    # test response for offer the same as the product
    data = {
        "product": items["product_ean_required"],
        "offer": items["offer_ean"]
    }
    response = test_app.post("/v1/match", data=json.dumps(data))
    response_body = response.json()

    assert response.status_code == 200
    assert response_body["match"] == "yes"
    assert response_body["details"] == "Ean match enabled and offer ean present among product eans."


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


# TODO: uncomment this when the name-matching will be allowed for selected categories
# def test_match_many_same(test_app, items):
#     # test response for offer the same as the product
#     data = {
#         "products": [items["product"], items["product"]],
#         "offer": items["offer_same"]
#     }
#     response = test_app.post("/v1/match-many", data=json.dumps(data))
#     response_body = response.json()

#     assert response.status_code == 200
#     assert response_body == [
#         {'match': 'yes', 'details': 'Name match: namesimilarity=1.0'},
#         {'match': 'yes', 'details': 'Name match: namesimilarity=1.0'}
#     ]


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
