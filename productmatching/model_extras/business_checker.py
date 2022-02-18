import os
from typing import List, Optional
from model_extras.unit_checker import UnitChecker, NumUnitCheck
from xgboostmatching.models import Decision, Match


class BusinessChecker():
    """
    Class containing the matching models' business logic which is independent of trained models.
    """
    def __init__(self, unit_conversions: bool = True):
        self.unit_checker = UnitChecker(unit_conversions)
        self.use_attribute_check = int(os.getenv("USE_ATTRIBUTE_CHECK", "0"))

    def primary_check(self, n_unmatched_attributes, norm_product_name, norm_offer_name, p_attrs=None, o_attrs=None):
        """
        This check precedes features creation and possible consequent XGB matching.
        Contains hard check for attributes and numerical units.
        Might only result in rejection or no action.
        """
        if self.use_attribute_check and n_unmatched_attributes > 0:
            details = f"Attributes mismatch: {n_unmatched_attributes}"
            if o_attrs is not None:
                details += f', item_attrs: {o_attrs}'
            if p_attrs is not None:
                details += f', candidate_attrs: {p_attrs}'
            return {
                "match": Match(
                    match=Decision.no,
                    details=details,
                ),
                "num_unit_check": None,
            }

        # check numerical units
        # direct contradiction (15kg vs 5kg) leads to rejection
        # indirect contradiction (15kg vs 'nothing') leads to change of 'yes' to 'unknown'
        # optional unit conversion based on param passed in kwargs
        # TODO: implement only for some categories
        num_unit_check = self.unit_checker(
            norm_product_name,
            norm_offer_name,
        )
        if num_unit_check.decision == 'no':
            return {
                "match": Match(
                    match=Decision.no,
                    details=f"Numerical unit mismatch - product: {num_unit_check.a_units}; offer: {num_unit_check.b_units}",
                ),
                "num_unit_check": num_unit_check,
            }
        return {
            "match": None,
            "num_unit_check": num_unit_check,
        }

    @staticmethod
    def secondary_check(
        namesimilarity: float, ean_feature: float, num_unit_check: NumUnitCheck,
        unique_names: bool = False, ean_required: bool = False, ns_upper_thresh: float = 1.01
    ):
        """
        This check follows features creation and precedes possible consequent XGB matching.
        Contains hard check for namesimilarity (in combination with numerical units) and ean.
        Might only result in pairing, 'unknown' or no action.
        """
        if unique_names:
            if namesimilarity > ns_upper_thresh:
                if num_unit_check.decision == 'unknown':
                    return Match(
                        match=Decision.unknown,
                        details=f"Decision YES, but possible numerical unit mismatch, namesimilarity={namesimilarity}",
                    )
                else:
                    return Match(
                        match=Decision.yes,
                        details=f"Name match: namesimilarity={namesimilarity}",
                    )

        if ean_required and ean_feature > 0:
            return Match(
                match=Decision.yes,
                details="Ean match enabled and offer ean present among product eans.",
            )

    @staticmethod
    def price_check(
        product_prices: Optional[List[float]], offer_price: float, a: float, b: float, c: float
    ):
        """
        This function REJECTS the matching if ratio of the largest price of a matched offer and
        the offer to be matched is too large.

        The rejection boundary is set to a/min(matched_offer_price, offer_price - b) + c,
        where `matched_offer_price` set to min or max of min of max of product_prices and offer prices.

        """
        if not product_prices or offer_price is None:
            return None

        if offer_price <= min(product_prices):
            minimal_price = min(min(product_prices), offer_price)
            maximal_price = max(min(product_prices), offer_price)
        elif offer_price >= max(product_prices):
            minimal_price = min(max(product_prices), offer_price)
            maximal_price = max(max(product_prices), offer_price)
        else:
            # at least two offers matched to a product exist with with price lower (and higher) than offer_price
            return None

        ratio = a/(minimal_price + b) + c

        frac = maximal_price / (minimal_price + 0.0000001)
        if frac > ratio + 0.0000001:
            return Match(
                match=Decision.no,
                details="price difference is too large"
            )
        else:
            return None
