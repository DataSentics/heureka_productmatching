import typing as t
from collections import defaultdict

from utilities.loader import Product
from utilities.normalize import normalize_string

PRODUCERS_SYNONYMS = ["vyrobce", "znacka"]


def extract_offers_attributes(
    offers: t.Iterable, fields: list = ["attributes", "parsed_attributes"], format: bool = False
) -> t.Union[defaultdict, dict]:
    """
    Merges information from fields/keys of all supplied offers into one dict.
    The values of the specified fields are expected to be lists of dicts that contain 'name' and 'value' fields.
    Merging means creating a set of all 'value's for each 'name'.

    :param offers: an iterable of dicts, should contain key(s) specified in `fields` param
    :param fields: dict keys to be merged together
    :param format: indicates, whether to return 'formatted' output = dict o lists instead of defaultdict of sets
    """
    attributes = defaultdict(set)
    for offer in offers:
        for attr_field in fields:
            for attr in (offer.get(attr_field) or []):
                name = normalize_string(str(attr.get("name", "")))
                value = normalize_string(str(attr.get("value", "")))
                if name and value:
                    if name in PRODUCERS_SYNONYMS:
                        name = "producer"
                    attributes[name].add(value)

    if format:
        # return dict with list values
        attributes = {n: list(v) for n, v in attributes.items()}

    return attributes


def parse_attributes(item: dict, mode: str = "name", field: str = "attributes") -> t.Dict[str, str]:
    assert mode in ["name", "unit"]
    attributes = item.get(field, [])

    if not attributes:
        return {}
    if mode == "name":
        parsed_atributes = {
            normalize_string(str(attr["name"])): normalize_string(str(attr["value"]))
            for attr in attributes
        }
        return {k: v for k, v in parsed_atributes.items() if k and v}
    else:
        res = defaultdict(list)
        for attr in attributes:
            if attr_unit := normalize_string(attr["unit"].strip()):
                if attr_value := normalize_string(attr["value"].strip()):
                    res[attr_unit].append(attr_value)
        return {k: sorted(v) for k, v in res.items()}


def get_collector_product_offers_attributes(args, product_id: t.Union[str, int], format: bool = False) -> t.Union[dict, defaultdict]:
    """
    We want to use this in order to prevent comparison of product attributes that might come from offer with the offer itself.

    There are two situations:
    - using collector data -> for each offer in test set, we have the info about its' paired product
        and for each product, we have info about all its offers
    - using static dataset -> for each offer in test set, we have the info about its' paired product
        for some products, we don't have direct acces to info about their offers
        but these products aren't correct match* for any of the offers, so the need to check the attributes should never arise
        * - if we assume the current pairing to be correct

    The default output type is defaultdict(defaultdict(list)) or dict(dict(list)) when `format` param is set to True.
    The output format is: {attribute_a_name: {attribute_a_value_1: [offer_ids], attribute_a_value_2:...}, attribute_b_name:...}
    """
    collector_offers_attributes = {}
    for offer in Product.offers(args.input_collector_offers, product_id):
        collector_offers_attributes[offer['id']] = extract_offers_attributes([offer])

    if not collector_offers_attributes:
        return {}

    # for each value and name, get a list of offers containing that attribute
    attributes_name_value_offers = defaultdict(lambda: defaultdict(list))
    for offer_id, attrs in collector_offers_attributes.items():
        for name, values in attrs.items():
            for value in values:
                attributes_name_value_offers[name][value].append(str(offer_id))
    if format:
        return {k: dict(v) for k, v in attributes_name_value_offers.items()}
    return attributes_name_value_offers


def rinse_product_attributes(offer_id: str, product_attributes: dict, product_attributes_name_value_offers: dict) -> dict:
    """
    Filter product attributes in a way such that those attributes present only in data of offer "offer_id" are not present.
    This is done in order to avoid comparing offers' attributes with its own attributes present among products' attributes.
    """
    product_attrs_new = {}
    for name, value in product_attributes.items():
        name_value_offers = set(product_attributes_name_value_offers.get(name, {}).get(value, []))
        # if the attribute is not present among paired offers or is present at some other offer than the one paired
        if (not name_value_offers) or (name_value_offers - {str(offer_id)}):
            product_attrs_new[name] = value

    return product_attrs_new
