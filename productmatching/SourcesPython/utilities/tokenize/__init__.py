from .tokenizers import WhiteSpaceTokenizer, BaseTokenizer
from typing import Union


TOKENIZERS = {
    'whitespacetokenizer': WhiteSpaceTokenizer,
}

def get_tokenizer(tokenizer: Union[str, BaseTokenizer]):
    if isinstance(tokenizer, BaseTokenizer):
        return tokenizer
    else:
        try:
            return TOKENIZERS[tokenizer.strip()]
        except KeyError:
            raise ValueError(f'At least one given tokenizer ({tokenizer}) does not exist. Should be on of {", ".join(TOKENIZERS.keys())}.')
