import ujson
import re
from pathlib import Path
from typing import Set, Union
from collections import defaultdict, namedtuple
from nltk import ngrams

from utilities.normalize import normalize_string

AttributeCheck = namedtuple("AttributeCheck", "decision a_attributes b_attributes")


class Attributes:
    def __init__(self, from_: Union[str, Path]):
        assert str(from_).endswith(".json"), "Invalid input file."

        self._category_to_name_to_values = defaultdict(lambda: defaultdict(set))
        self._category_to_value_to_names = defaultdict(lambda: defaultdict(set))
        self._category_to_value_to_synonym = defaultdict(lambda: defaultdict(set))

        with open(from_, "r") as attrfile:
            category_to_name = ujson.load(attrfile)

        for category in category_to_name:
            for name, synonyms in category_to_name[category].items():
                for value_synonyms in synonyms:
                    if not type(value_synonyms) == list:
                        value_synonyms = [value_synonyms]
                    for value in value_synonyms:
                        # dont use attributes with values like "5-10", "15%", ... - might be ambiguous
                        if self.check_nonum_val(value):
                            value = value.strip()

                            self._category_to_value_to_names[category][value].add(name)
                            self._category_to_value_to_synonym[category][value].update(value_synonyms)
                            self._category_to_name_to_values[category][name].add(value)

        self._category_to_name_to_values.default_factory = None
        self._category_to_value_to_names.default_factory = None
        self._category_to_value_to_synonym.default_factory = None

    @staticmethod
    def check_nonum_val(value):
        num_val_regex = "^[ 0-9.,-_%*+]+$"
        if re.match(num_val_regex, value):
            return False
        return True

    def names(self, value: str, category: str) -> Set[str]:
        if category not in self._category_to_value_to_names:
            return set()

        return self._category_to_value_to_names[category][value]

    def values(self, name: str, category: str) -> Set[str]:
        if category not in self._category_to_value_to_names:
            return set()

        return self._category_to_name_to_values[category][name]

    def _get_attributes(self, s: str, category: str, n_grams: int) -> dict:
        _grams = self._get_grams(s, n_grams)
        attributes = defaultdict(set)
        if category not in self._category_to_value_to_names:
            return attributes
        for _gram in _grams:
            # not original or edited - no possible clue
            if _gram not in self._category_to_value_to_names[category]:
                continue
            attr_names = self._category_to_value_to_names[category][_gram]

            _gram_synonyms = set()
            # one of the original values/synonyms
            if self._category_to_value_to_synonym[category].get(_gram):
                _gram_synonyms = self._category_to_value_to_synonym[category][_gram]

            for name in attr_names:
                attributes[name] |= _gram_synonyms
        return attributes

    def _get_grams(self, s: str, n_grams: int) -> set:
        grams = set()
        for n in range(1, n_grams + 1):
            grams.update(" ".join(x) for x in ngrams(s.split(" "), n))
        return grams

    def check_names(
        self,
        a: str,
        b: str,
        category_id: str,
        normalized: bool = True,
        soft: bool = True,
    ) -> AttributeCheck:
        if not normalized:
            a, b = normalize_string(a), normalize_string(b)

        if category_id in self._category_to_value_to_names:
            n_grams = max([len(value.split(' ')) for value in self._category_to_value_to_names[category_id].keys()])

            a_attributes = self._get_attributes(a, category_id, n_grams)
            b_attributes = self._get_attributes(b, category_id, n_grams)
        else:
            a_attributes = {}
            b_attributes = {}

        def compare_sets(s1: set, s2: set, soft: bool) -> bool:
            if soft:
                return bool(s1 & s2)
            else:
                return (s1.issubset(s2) or s2.issubset(s1))

        # this is the only case when we want to return False!
        # a_attributes = {'A': {'x'}, 'B': {'b'}}, b_attributes = {'A': {'y'}, 'C': {'c'}}
        if any([
            not compare_sets(a_attributes[k], b_attributes[k], soft)
            for k in set(a_attributes.keys()) & set(b_attributes.keys())
        ]):
            return AttributeCheck(False, a_attributes, b_attributes)

        # all other possible cases are considered to be fine:
        # a_attributes = {'A': {'a'}, 'B': {'b'}}, b_attributes = {'A': {'a'}, 'C': {'c'}}
        # a_attributes = {'A': {'a'}, 'B': {'b'}}, b_attributes = {'A': {'a'}}
        # a_attributes = {'A': {'a'}}, b_attributes = {'A': {'a'}, 'C': {'c'}}
        # a_attributes = {'A': {'a'}}, b_attributes = {'A': {'a'}}
        # a_attributes = {'A': {'a'}}, b_attributes = {'C': {'c'}}
        # a_attributes == {} or b_attributes == {}

        return AttributeCheck(True, a_attributes, b_attributes)

    def contain(self, value, category_id):
        return category_id in self._category_to_value_to_names and value in self._category_to_value_to_names[category_id]
