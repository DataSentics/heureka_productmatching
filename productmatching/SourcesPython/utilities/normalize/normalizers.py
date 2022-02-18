import re
import unidecode
import typing as t

import nltk

from .stop_words import STOP_WORDS
from .regexes import get_regexes
from .internal import REPLACE_OF_WITH, STRIPPERS
from ..config_pipeline import PREPROCESS_CONFIG
from ..args import convert_at_args


class BaseNormalizer:

    def _split_input(self, X: t.Union[str, t.Iterable[str]]) -> t.List[str]:
        """ Return list of tokens """
        if isinstance(X, str):
            return X.split(" ")
        elif isinstance(X, t.Iterable):
            return list(X)
        raise ValueError(f"Wrong type of input: {type(X)}")


    def _concatenate_input(self, X: t.Union[str, t.Iterable[str]]) -> str:
        """ Return space separated tokens in string """
        if isinstance(X, str):
            return X
        elif isinstance(X, t.Iterable):
            return " ".join(X)
        raise ValueError(f"Wrong type of input: {type(X)}")


    def _apply_kwargs(self, **kwargs):
        pass


    def __call__(self, X: t.Union[str, t.Iterable[str]], return_list:bool =False) -> t.Union[str, t.List[str]]:
        """ Given a string of space separated tokens or an iterable of tokens, normalize it.
        It can be callable either pre or post tokenization.

        Args:
            X (str, Iterable[str]): a space separated tokens or an iterable of tokens

        Returns:
            (str or list[str]): normalized input
        """
        raise NotImplementedError()


    def __str__(self):
        raise NotImplementedError()


    @classmethod
    def from_config(cls, config: t.Union[str, dict], data_directory:str = None, **kwargs):
        n = cls()
        if isinstance(config, str):
             config = convert_at_args(config, data_directory)
        conf = PREPROCESS_CONFIG['configs'][str(n)]
        conf.update(config)
        conf.update(kwargs)
        n._apply_kwargs(**conf)
        return n


class DoNothingNormalizer(BaseNormalizer):
    def __call__(self, X, return_list:bool = False):
        if not return_list:
            return self._concatenate_input(X)
        else:
            return self._split_input(X)

    @classmethod
    def from_config(cls, config: t.Union[str, dict], data_directory:str = None, **kwargs):
        return cls()

    def __str__(self):
        return 'donothingnormalizer'


class OriginalPmiNormalizer(BaseNormalizer):

    def _apply_kwargs(self, **kwargs):
        pmi_file = kwargs.get('pmi_file', kwargs['input_pmi'])  # either one has to exist
        unit_conversions = kwargs.get('unit_conversions', True)

        self.unit_conversions = unit_conversions
        self.regexes, self.conversion_regex = get_regexes(unit_conversions)

        try:
            with open(pmi_file, "r") as f:
                coocurences = f.read().split("\n")
        except FileNotFoundError:
            coocurences = []

        self.coocurences = set(tuple(c.strip().split(" ")) for c in coocurences if c.strip() != "")
        self.n_gram = len(coocurences[0].split(" ")) if len(self.coocurences) else 0


    def __call__(self, X, return_list:bool = False):
        """ The `return_list` is just for keeping a common interface, it does not do anything.
        A string is returned always. """

        X = self._concatenate_input(X)
        X = normalize_string(X, False, self.regexes, self.conversion_regex, self.unit_conversions)

        if self.n_gram > 0:
            for gram in nltk.ngrams(self._split_input(X), self.n_gram):
                if gram in self.coocurences:
                    X = X.replace(" ".join(gram), "_".join(gram))

        return X


    @classmethod
    def from_config(cls, config: t.Union[str, dict], data_directory:str = None, **kwargs):
        n = super().from_config(config)
        return n


    def __str__(self):
        return 'originalpminormalizer'


def normalize_string(string: str, return_list: bool = False, regexes=None, conversion_regex=None, unit_conversions: bool = False):
    if not string:
        return ""
    # get regexes if not passed in params
    if not regexes:
        regexes, conversion_regex = get_regexes(unit_conversions)

    # replace prefixes in title, e.g. Fitmin 0.5kg -> Fitmin 0.5kilog
    # title is then lowercased and we would not identify e.g. mili (m) and mega (M)
    # used only in unitchecker for now
    if unit_conversions:
        string = re.sub(conversion_regex[0], conversion_regex[1], string)

    # lowercase, unit prefixes are processed or ignored
    string = string.lower()
    string = unidecode.unidecode(string)

    for (of, replacement) in REPLACE_OF_WITH:
        string = string.replace(of, replacement)

    for (regex, replacement) in regexes:
        string = re.sub(regex, replacement, string)

    stripped = [word.strip(STRIPPERS) for word in string.split(" ")]

    added = set()
    result = "" if not return_list else []

    for word in stripped:
        if not word:
            continue

        if word in STOP_WORDS:
            continue

        if word in added:
            continue

        if len(added) > 0 and not return_list:
            result += " "

        added.add(word)
        result += word if not return_list else [word]

    return result


