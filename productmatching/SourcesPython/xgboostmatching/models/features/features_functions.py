import typing as t
from collections import defaultdict

from utilities.normalize import normalize_string
from utilities.damerau_levenshtein import damerau_levenshtein_one
from utilities.priceoutlier.priceoutlier import priceoutlier

from .item_classes import Product, Offer

# All the fuctions should accept only `product`, `offer` and `**kwargs` as an input
# All the fuctions should return a tuple or a list


def match_shop(
    product: Product,
    offer: Offer,
    **kwargs
) -> t.List[float]:
    # for delisted products
    if not product.shops:
        return [0]
    else:
        return [float(sum(s == offer.shop for s in product.shops))]


def match_ean(
    product: Product,
    offer: Offer,
    **kwargs
) -> t.List[float]:
    if offer.ean is None or not product.eans:
        return (0, 0)
    else:
        return (1, float(offer.ean in product.eans))


def is_sub_name(
    product: Product,
    offer: Offer,
    **kwargs
) -> t.Tuple[float, float]:
    namesimilarity = kwargs.get("namesimilarity")
    if namesimilarity is not None:
        pipeline = namesimilarity.pipeline

    use_dl = kwargs.get("use_dl", True)
    # 'sub name' definition: set of words in a first name
    # is a subset of the second names' set of words

    # when there is a multiple occurence of one word in a name
    # we consider it to be just a case of improper naming
    # this means that "A A B" is considerd to be contained in "A B C"

    # normalized names
    p_norm = pipeline(product.name)
    o_norm = pipeline(offer.name)

    # sets of words in names
    p_name = set(p_norm.split(' '))
    o_name = set(o_norm.split(' '))

    if not use_dl:
        return (float(p_name.issubset(o_name)), float(o_name.issubset(p_name)))

    # Sometimes, a name can contain a typo, we try to correct the typo in one name using dam.-lev. distance 1
    # this is considered 'weaker' then direct sub name identification

    # sets of sets of words in names in DL distance 1 form original normalized name
    # we check the subset property of p/o_name for each set contained in o/p_name_dl
    # The p/o_name is in such case considered to be correct and we look for typos in the other name
    # This means, that "pes kocka" vs. "pes kocka koceka" will result in reporting of no subset
    p_name_dl = [set(p_dl.split(' ')) for p_dl in damerau_levenshtein_one(p_norm) - {p_norm}]
    o_name_dl = [set(o_dl.split(' ')) for o_dl in damerau_levenshtein_one(o_norm) - {o_norm}]

    if p_name.issubset(o_name):
        # product name is a direct subset of offer name
        p_res = 2
    elif any([p_name.issubset(o_dl) for o_dl in o_name_dl]):
        # product name is a subname of some augmented offer name
        p_res = 1
    else:
        p_res = 0

    if o_name.issubset(p_name):
        # offer name is a direct subset of product name
        o_res = 2
    elif any([o_name.issubset(p_dl) for p_dl in p_name_dl]):
        # offer name is a subname of some augmented product name
        o_res = 1
    else:
        o_res = 0

    return (float(p_res), float(o_res))


def match_nameattributes(
    product: Product,
    offer: Offer,
    **kwargs
):
    attributes = kwargs.get("attributes")
    nameattributes_matched = 0
    nameattributes_unmatched = 0
    check = attributes.check_names(product.name, offer.name, product.category_id, False)
    p_attrs, o_attrs = check.a_attributes, check.b_attributes
    for name in set(o_attrs.keys()) & set(p_attrs.keys()):
        if o_attrs[name] & p_attrs[name]:
            nameattributes_matched += 1
        else:
            nameattributes_unmatched += 1

    namu = nameattributes_matched + nameattributes_unmatched
    if namu == 0:
        return (0, 0)
    else:
        return (
            # common nameattributes indicator
            float(namu > 0),
            # % of matched nameattributes
            nameattributes_matched / namu,  # ratio
        )


def match_attributes(
    product: Product,
    offer: Offer,
    **kwargs
) -> t.Tuple[int, int, int, float, dict, dict]:
    external_match_attributes_result = kwargs.get("external_match_attributes_result", {})
    if external_match_attributes_result:
        return external_match_attributes_result
    # these are likely not to be used
    product_nameattributes = kwargs.get("product_nameattributes", {})
    offer_nameattributes = kwargs.get("offer_nameattributes", {})
    # the sets are used just to be sure
    product_attributes = defaultdict(set)
    offer_attributes = defaultdict(set)
    for name, value in product.attributes.items():
        product_attributes[normalize_string(name)].add(normalize_string(value))
    for name, values in product_nameattributes.items():
        product_attributes[name] |= values

    for name, value in offer.attributes.items():
        if type(value) == str:
            offer_attributes[normalize_string(name)] |= {normalize_string(value)}
        else:
            offer_attributes[normalize_string(name)] |= {normalize_string(val) for val in value}

    for name, value in offer.parsed_attributes.items():
        if type(value) == str:
            offer_attributes[normalize_string(name)] |= {normalize_string(value)}
        else:
            offer_attributes[normalize_string(name)] |= {normalize_string(val) for val in value}

    for name, values in offer_nameattributes.items():
        offer_attributes[name] |= values

    # match ~ product_attributes[attr] & offer_attributes[attr] != set()
    matched, unmatched = 0, 0

    for p_name, p_atr_values in product_attributes.items():
        if p_name not in offer_attributes:
            continue

        if p_atr_values & offer_attributes[p_name]:
            matched += 1

        else:
            unmatched += 1

    p_attrs_formated = {k: list(v) for k, v in product_attributes.items() if k in offer_attributes}
    o_attrs_formated = {k: list(v) for k, v in offer_attributes.items() if k in product_attributes}

    amu = matched + unmatched

    if amu == 0:
        return (
            *((0,) * 4),
            p_attrs_formated,
            o_attrs_formated
        )
    else:
        return (
            1,  # indicator
            amu,  # n common attributes
            matched,  # n common matched attributes
            float(matched / amu),  # % common matched attributes
            p_attrs_formated,
            o_attrs_formated
        )


def get_namesimilarity_result(
    product: Product,
    offer: Offer,
    **kwargs
):
    namesimilarity = kwargs.get("namesimilarity")
    namesimilarity_result = namesimilarity((product.name, offer.name))
    return namesimilarity_result,


def get_priceoutlier_result(
    product: Product,
    offer: Offer,
    **kwargs
):
    priceoutlier_result = priceoutlier(
        value=offer.price,
        data=product.prices,
        const=1.25,
        significance=0.1,
    )
    return (
        priceoutlier_result.tests.constant.value,
        priceoutlier_result.tests.bartlett,
    )
