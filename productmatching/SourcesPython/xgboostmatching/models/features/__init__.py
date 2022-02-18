from itertools import chain

from .item_classes import Product, Offer
from .features_conf import (
    FEATURES_CONFIG,
    numerical_features,
    discrete_features,
    all_features,
    NAMESIMILARITY_NAME
)
from .features_functions import match_attributes


async def _general_feature_calculation(hlf_conf: dict, product: Product, offer: Offer, **kwargs):
    if not hlf_conf:
        return

    if hlf_conf.get("async", False):
        result = await hlf_conf["function"](product, offer, **kwargs)
    else:
        result = hlf_conf["function"](product, offer, **kwargs)

    # extracting features from the output of `hlf_conf["function"]`,
    # returning list of tuples containing (feature_name, feature_value)
    return [(name, result[index]) for feature_conf in hlf_conf["features"] for name, index in feature_conf["names_indexes"]]


async def create(
    product: Product,
    offer: Offer,
    selected_features: dict = None,
    **kwargs
) -> list:
    f_conf = FEATURES_CONFIG
    # calculate only selected features
    if selected_features:
        f_conf = {k: v for k, v in FEATURES_CONFIG.items() if k in selected_features}
        for hl_feature, features_idxs in selected_features.items():
            if hl_feature in f_conf:
                f_conf[hl_feature]["features"] = [f_conf[hl_feature]["features"][i] for i in features_idxs]

    results = list(chain(*[
        await _general_feature_calculation(hln_conf, product, offer, **kwargs) for hln_conf in f_conf.values()
    ]))

    if kwargs.get("names", False):
        return results

    return [r[1] for r in results]
