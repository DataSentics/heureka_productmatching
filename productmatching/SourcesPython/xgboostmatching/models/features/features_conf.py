from .features_functions import (
    match_shop,
    match_ean,
    is_sub_name,
    match_nameattributes,
    match_attributes,
    get_priceoutlier_result,
    get_namesimilarity_result,
)

NAMESIMILARITY_F_CONF_NAME = "get_namesimilarity_result"
NAMESIMILARITY_NAME = "namesimilarity"


# following config contains essential information on all featrues for xgboost model
# the structure is following:
# On the highest level are key-value pairs representing features created by one function from `.features_functions`, e.g. price features
# the value dict should contain info on features, that should always 'stick together' and contains following information:
# - function: the function used for creation of corresponding features
# - async: boolean value telling whether the function ^ is asynchronous, absence is equivalent to `False`
# - features: list of dicts containing detailed information on the features themselves, each member of the list contains the following info:
#   - names_indexes: list of tuples of features' names and indexes on which to look for the feature in the output of the function
#   - type: must be one of ["discrete", "numerical"], indirectly used in `DataVisualiser` class

# in order to disable certain features, comment either:
# - a full key-value pair of FEATURES_CONFIG, or
# - a member of chosen `features` list, or
# - a member of chosen `features` lists' `names_indexes` list
FEATURES_CONFIG = {
    "match_shop": {
        "function": match_shop,
        "features": [
            {
                "names_indexes": [("shops", 0)],
                "type": "discrete",
            },
        ]
    },
    "match_ean": {
        "function": match_ean,
        "features": [
            {
                "names_indexes": [
                    ("eans_not_nan", 0),
                    ("offer_ean_in_products_eans", 1)
                ],
                "type": "discrete",
            },
        ]
    },
    "is_sub_name": {
        "function": is_sub_name,
        "features": [
            {
                "names_indexes": [
                    ("pname_subset_oname", 0),
                    ("oname_subset_pname", 1)
                    ],
                "type": "discrete",
            },
        ]
    },
    "match_nameattributes": {
        "function": match_nameattributes,
        "features": [
            {
                "names_indexes": [("i_nameattributes", 0)],
                "type": "discrete",
            },
            {
                "names_indexes": [("r_nameattributes", 1)],
                "type": "numerical",
            },
        ]
    },
    "match_attributes": {
        "function": match_attributes,
        "features": [
            {
                "names_indexes": [("i_attributes", 0)],
                "type": "discrete",
            },
            {
                "names_indexes": [("r_attributesmatched", 3)],
                "type": "numerical",
            },
        ]
    },
    "get_priceoutlier_result": {
        "function": get_priceoutlier_result,
        "features": [
            {
                "names_indexes": [("constant", 0)],
                "type": "discrete",
            },
            {
                "names_indexes": [("bartlett", 1)],
                "type": "numerical",
            },
        ]
    },
    NAMESIMILARITY_F_CONF_NAME: {
        "function": get_namesimilarity_result,
        "features": [
            {
                "names_indexes": [(NAMESIMILARITY_NAME, 0)],
                "type": "numerical",
            },
        ]
    },
}

numerical_features = [
    name for feats in FEATURES_CONFIG.values() for fc in feats["features"] for name, index in fc["names_indexes"] if fc["type"] == "numerical"
]

discrete_features = [
    name for feats in FEATURES_CONFIG.values() for fc in feats["features"] for name, index in fc["names_indexes"] if fc["type"] == "discrete"
]

all_features = discrete_features + numerical_features
