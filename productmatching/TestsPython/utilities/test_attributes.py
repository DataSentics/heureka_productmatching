import pytest
import sys

from pathlib import Path

sys.path.append("/app/SourcesPython")
from utilities.attributes import Attributes, AttributeCheck


DATASET_PATH = Path(__file__).resolve().parents[1] / 'dataset' / 'utilities'
ATTRIBUTES_FILE = DATASET_PATH / 'attributes.json'


@pytest.fixture
def attributes():
    attr = Attributes(str(ATTRIBUTES_FILE))
    return attr


def test_attribute(attributes):
    result = attributes.check_names("apple iphone new",
                                    "samsung neco new",
                                    category_id="2000")

    assert result == AttributeCheck(True,
                                    a_attributes={'keywords': {'new'}},
                                    b_attributes={'keywords': {'new'}})


def test_without_attributes(attributes):
    result = attributes.check_names("apple iphone",
                                    "samsung neco",
                                    category_id="2000")

    assert result == AttributeCheck(True,
                                    a_attributes={},
                                    b_attributes={})


def test_true_typo_atribute(attributes):
    result = attributes.check_names("apple iphone grain new",
                                    "samsung neco niw",
                                    category_id="2000")

    assert result == AttributeCheck(True,
                                    a_attributes={'keywords': {'new'}},
                                    b_attributes={})


def test_false_typo_atribute(attributes):
    result = attributes.check_names("apple iphone grain new",
                                    "samsung neco now",
                                    category_id="2000")

    assert result == AttributeCheck(False,
                                    a_attributes={'keywords': {'new'}},
                                    b_attributes={'keywords': {'now'}})


def test_synonym_atribute(attributes):
    result = attributes.check_names("apple iphone glutan free",
                                    "samsung neco gf",
                                    category_id="2000")

    assert result == AttributeCheck(True,
                                    a_attributes={'keywords': {'glutan free', 'gf'}},
                                    b_attributes={'keywords': {'glutan free', 'grain free', 'gf'}})


def test_false_attribute(attributes):
    result = attributes.check_names("apple iphone grain free",
                                    "samsung neco glutan free",
                                    category_id="2000", soft=False)

    assert result == AttributeCheck(False,
                                    a_attributes={'keywords': {'grain free', 'gf'}},
                                    b_attributes={'keywords': {'glutan free', 'gf'}})


def test_false_attribute_soft(attributes):
    result = attributes.check_names("apple iphone grain free",
                                    "samsung neco glutan free",
                                    category_id="2000")

    assert result == AttributeCheck(True,
                                    a_attributes={'keywords': {'grain free', 'gf'}},
                                    b_attributes={'keywords': {'glutan free', 'gf'}})


def test_attribute_from_different_categories(attributes):
    result = attributes.check_names("apple iphone bezna",
                                    "samsung neco premium",
                                    category_id="1800")

    assert result == AttributeCheck(True,
                                    a_attributes={},
                                    b_attributes={'dogfood': {'premium'}})


def test_several_attributes_in_one_name(attributes):
    result = attributes.check_names("apple iphone male new",
                                    "samsung neco new",
                                    category_id="2000")

    assert result == AttributeCheck(True,
                                    a_attributes={"keywords": {'new'}, "velikost plemene": {'male'}},
                                    b_attributes={"keywords": {'new'}})


def test_unknown_category(attributes):
    result = attributes.check_names("apple iphone male new",
                                    "samsung neco new",
                                    category_id="6666")

    assert result == AttributeCheck(True,
                                    a_attributes={},
                                    b_attributes={})
