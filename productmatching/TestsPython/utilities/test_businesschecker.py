import sys
sys.path.append("/app/SourcesPython")

import pytest

from model_extras.business_checker import BusinessChecker
from xgboostmatching.models import Decision

# TODO fix all this

def test_price_ok():
    price_reject_a = 1000.0
    price_reject_b = 400.0
    price_reject_c = 2.5
    product_prices = [100.0,110.0,120.0,130.0]
    offer_price = 100.0
    result = BusinessChecker.price_check(product_prices, offer_price, price_reject_a, price_reject_b, price_reject_c)
    assert result is None


def test_no_product_prices():
    price_reject_a = 1000.0
    price_reject_b = 400.0
    price_reject_c = 2.5
    product_prices = None
    offer_price = 100.0
    result = BusinessChecker.price_check(product_prices, offer_price, price_reject_a, price_reject_b, price_reject_c)
    assert result is None


def test_offer_lower_but_accept():
    price_reject_a = 1000.0
    price_reject_b = 400.0
    price_reject_c = 2.5
    product_prices = [100.0,110.0,120.0,130.0]
    offer_price = 70.0
    result = BusinessChecker.price_check(product_prices, offer_price, price_reject_a, price_reject_b, price_reject_c)
    assert result is None


def test_offer_lower_reject():
    price_reject_a = 1000.0
    price_reject_b = 400.0
    price_reject_c = 2.5
    product_prices = [1000.0,1100.0,1200.0,1300.0]
    offer_price = 200.0
    result = BusinessChecker.price_check(product_prices, offer_price, price_reject_a, price_reject_b, price_reject_c)
    assert result.match == Decision.no

def test_offer_higher_reject():
    price_reject_a = 1000.0
    price_reject_b = 400.0
    price_reject_c = 2.5
    product_prices = [1000.0,1100.0,1200.0,1300.0]
    offer_price = 6000.0
    result = BusinessChecker.price_check(product_prices, offer_price, price_reject_a, price_reject_b, price_reject_c)
    assert result.match == Decision.no

def test_offer_higher_accept():
    price_reject_a = 1000.0
    price_reject_b = 400.0
    price_reject_c = 2.5
    product_prices = [20000.0,21100.0,21200.0,21300.0]
    offer_price = 26000.0
    result = BusinessChecker.price_check(product_prices, offer_price, price_reject_a, price_reject_b, price_reject_c)
    assert result is None


def test_offer_higher_reject2():
    price_reject_a = 1000.0
    price_reject_b = 400.0
    price_reject_c = 2.5
    product_prices = [20000.0,21100.0,21200.0,21300.0]
    offer_price = 60000.0
    result = BusinessChecker.price_check(product_prices, offer_price, price_reject_a, price_reject_b, price_reject_c)
    assert result.match == Decision.no


def test_offer_higher_but_accept():
    price_reject_a = 1000.0
    price_reject_b = 400.0
    price_reject_c = 2.5
    product_prices = [100.0,110.0,120.0,130.0]
    offer_price = 250.0
    result = BusinessChecker.price_check(product_prices, offer_price, price_reject_a, price_reject_b, price_reject_c)
    assert result is None


def test_offer_higher_reject3():
    price_reject_a = 1000.0
    price_reject_b = 400.0
    price_reject_c = 2.5
    product_prices = [100.0,110.0,120.0,130.0]
    offer_price = 1440.0
    result = BusinessChecker.price_check(product_prices, offer_price, price_reject_a, price_reject_b, price_reject_c)
    assert result.match == Decision.no


def test_offer_diff_accept():
    price_reject_a = 1000.0
    price_reject_b = 400.0
    price_reject_c = 2.5
    product_prices = [10.0,11.0,12.0,13.0]
    offer_price = 50
    result = BusinessChecker.price_check(product_prices, offer_price, price_reject_a, price_reject_b, price_reject_c)
    assert result is None
