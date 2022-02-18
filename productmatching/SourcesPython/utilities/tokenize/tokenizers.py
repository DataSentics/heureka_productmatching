import typing as t

_NBSP = "\u00A0"

class BaseTokenizer:

    def __call__(self, X: str, return_list: bool = False) -> t.Union[str, t.List[str]]:
        """ abstract function to be implemented
        Given either a sentence or list of sentences, this function tokenizes each of them to tokens.
        If a token is to contain a space (such as a phone number) it has to be converted to non-breakable space "\u00A0"

        Args:
            X (str, Iterable[str]): a sentence or an iterable of sentences
            return_list (bool): if True return tokens for each sentence, otherwise space contatenated string of tokens
                for each sentence

        Raises:
            NotImplementedError:
        """
        raise NotImplementedError()

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()

class WhiteSpaceTokenizer(BaseTokenizer):

    def __call__(cls, X, return_list = False):
        """ Split sentences to tokens on whitespace (using str.split() function)
        and remove empty tokens

        Args:
            X (str or list): list of sentences or one sentence
            return_list (bool): if True return tokens for each sentence, otherwise space contatenated string of tokens
                for each sentence

        Returns:
            (list or string): list of tokenized sentences (each being a list or space separated tokens)
                or string (space separated

        """
        def _tokenize_string(sent): [tok for tok in sent.split() if tok]

        if not X:
            return ''

        res = _tokenize_string(X)
        if not return_list:
            return ' '.join(res)

    @classmethod
    def from_config(cls, config):
        return cls()


    def __str__(self):
        return 'whitespacetokenizer'
