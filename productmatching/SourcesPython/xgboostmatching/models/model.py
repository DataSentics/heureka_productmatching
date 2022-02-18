import os
import logging
import asyncio
import numpy as np
import typing as t

import pandas as pd
import xgboost as xgb

from model_extras.business_checker import BusinessChecker
from preprocessing.models import exact_match
from utilities.normalize.normalizers import OriginalPmiNormalizer
from utilities.attributes import Attributes
from xgboostmatching.models import features, Decision, Match


class XGBoostMatchingModel:
    def __init__(
        self,
        xgboost_path: str,
        namesimilarity: exact_match.ExactMatchModel,
        attributes: Attributes,
        tok_norm_args: t.Union[str, dict],
        thresholds_path: str = None,
        feature_oc: bool = False,
        unit_conversions: bool = True,
        price_reject_a: float = 1000.0,
        price_reject_b: float = 400.0,
        price_reject_c: float = 2.5
    ):
        self.booster = xgb.Booster()
        self.namesimilarity = namesimilarity
        self.attributes = attributes
        self.business_checker = BusinessChecker(unit_conversions)
        self.booster.load_model(xgboost_path)
        self.booster.feature_names = self.booster.attr('feature_names').split('|')
        self.default_thresholds = (0.45, 0.50)
        self.thresholds = self.get_thresholds(thresholds_path)
        self.namesimilarity_upper_threshold = 1.01

        self.price_reject_a = price_reject_a
        self.price_reject_b = price_reject_b
        self.price_reject_c = price_reject_c

        self.feature_oc = feature_oc

        self.original_pmi_normalizer = OriginalPmiNormalizer.from_config(tok_norm_args, unit_conversions=unit_conversions)

    def get_thresholds(self, thresholds_path):
        if (thresholds_path is None) or (not os.path.isfile(thresholds_path)):
            # use default thresholds
            logging.info(f"Using default thresholds {self.default_thresholds}")
            return self.default_thresholds
        else:
            with open(thresholds_path, "r") as f:
                thresholds = f.read()
            logging.info(f"Read thresholds {thresholds} from {thresholds_path}")
            return tuple(float(t) for t in thresholds.split(","))

    def predict(self, input):
        product = input['product']
        offer = input['offer']
        # method is called only on matchapi where running an async model from mlflow seems problematic
        # use not async code in matchapi and RegistryMatchingModelConstructor and run a synthetic asyncio here
        result = asyncio.run(self.__call__(product, offer))
        return result

    def get_features_contribution(self, data, max_n_features: int = 8):
        """Calculates contribution of features to prediction using shap values."""
        # calculate shap values for local interpretation
        # using implementation in xgboost, the same as if using shap module
        local_shap = self.booster.predict(data, pred_contribs=True)[0]
        # extract base (expected) value
        base_value = local_shap[-1]
        # shap values for prediction
        local_shap_v = local_shap[:-1]

        order = np.argsort(-1*np.abs(local_shap_v))
        # limit number of most influential features
        n_to_select = min(max_n_features, np.count_nonzero(local_shap_v))
        # get selected feature names and shap values
        sel_shap_v = local_shap_v[order][:n_to_select]
        sel_feature_names = np.array(self.booster.feature_names)[order][:n_to_select]

        # format as string in format (shap1*feature1)+(shap2*feature2)+..., round shap values
        contributions = "+".join(
            [
                f"({contr:.3f}*{feature})"
                for feature, contr in
                zip(sel_feature_names, sel_shap_v)
            ]
        )
        # add base base value
        contributions = contributions + f"+({base_value:.3f}*base_value)"

        return contributions

    async def __call__(
        self,
        product: features.Product,
        offer: features.Offer,
    ) -> Match:
        # check attributes in data
        # reject the pair if any of the common attributes differs
        # TODO: possibly change this in future when some more reliable way of atributes comparison is available
        attributes_check = features.match_attributes(product, offer)
        n_unmatched_attributes = attributes_check[1] - attributes_check[2]  # n_common_attributes - n_matched_attributes

        price_check = self.business_checker.price_check(product.prices, offer.price, self.price_reject_a, self.price_reject_b, self.price_reject_c)
        if price_check is not None:
            return price_check

        # using explicitly the OriginalPmiNormalizer because unit checker heavily depends on it.
        norm_product_name = self.original_pmi_normalizer(product.name)
        norm_offer_name = self.original_pmi_normalizer(offer.name)

        primary_check = self.business_checker.primary_check(
            n_unmatched_attributes, norm_product_name, norm_offer_name, attributes_check[3], attributes_check[4]
        )
        if primary_check["match"] is not None:
            return primary_check["match"]

        features_ = await features.create(
            product=product,
            offer=offer,
            namesimilarity=self.namesimilarity,
            attributes=self.attributes,
            names=True,
            feature_oc=self.feature_oc,
            external_match_attributes_result=attributes_check,
        )
        names, vals = zip(*features_)
        features_s = pd.Series(vals, names)

        ns = features_s[features.NAMESIMILARITY_NAME]
        ean_f = 0
        if 'offer_ean_in_products_eans' in features_s.index:
            ean_f = features_s['offer_ean_in_products_eans']

        secondary_check = self.business_checker.secondary_check(
            ns, ean_f, primary_check["num_unit_check"], product.unique_names, product.ean_required, self.namesimilarity_upper_threshold
        )
        if secondary_check is not None:
            return secondary_check

        data = xgb.DMatrix(
            np.array([features_s[self.booster.feature_names]]),
            feature_names=self.booster.feature_names
        )
        pred = self.booster.predict(data)[0]

        if pred < self.thresholds[0]:
            decision = Decision.no

        elif pred < self.thresholds[1]:
            decision = Decision.unknown

        else:
            decision = Decision.yes

        if decision == Decision.yes and primary_check["num_unit_check"].decision == 'unknown':
            return Match(
                match=Decision.unknown,
                details=f"Decision YES, but possible numerical unit mismatch, namesimilarity={ns}, prediction={pred}",
            )

        # feature contribution using shap values for observation
        contributions = self.get_features_contribution(data)
        # feature values
        feature_values = [f"{fn}={val}" for fn, val in features_ if fn in features.all_features]

        return Match(
            match=decision,
            details=",".join(
                [f"prediction={pred}"] + [contributions] + feature_values
            ),
            confidence=pred,
        )
