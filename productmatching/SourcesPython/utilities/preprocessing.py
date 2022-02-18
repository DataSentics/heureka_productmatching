import typing as t

from .normalize import get_normalizer, NORMALIZERS
from .tokenize import get_tokenizer, TOKENIZERS
from .normalize import *
from .config_pipeline import PREPROCESS_CONFIG
from .args import convert_at_args


class Pipeline:

    def __init__(self):
        self.config = None
        self.processor_configs = {}

    @classmethod
    def create(cls, tok_norm_args: t.Union[dict, str] = None, data_directory: str = None, **kwargs):
        """ Create a new pipeline.
            `tok_norm_args` update the default config and `kwargs` update tok_norm_args

        Args:
            tok_norm_args (dict or str): arguments for both tokenizers and normalizers
            data_directory (str): path to where possible artifacts are

        Returns:
            new instance
        """
        p = cls()
        config = PREPROCESS_CONFIG.copy()
        if isinstance(tok_norm_args, str):
            tok_norm_args_dict = convert_at_args(tok_norm_args, data_directory)
        else:
            tok_norm_args_dict = tok_norm_args
        p.config = config
        p.processor_configs = config['configs']  # dict
        p._init_from_processors(config['processors'], tok_norm_args_dict, kwargs)
        return p


    def _init_from_processors(self, processors: t.List[str], tok_norm_args_dict: t.Optional[dict], _kwargs: t.Optional[dict]):
        self.processors = []
        for p in processors:
            # default config is handled inside the classes; tok_norm_args_dict overrides common keys
            p_config = PREPROCESS_CONFIG['configs'][str(p)]
            if tok_norm_args_dict:
                p_config.update(tok_norm_args_dict)
            if _kwargs:
                p_config.update(_kwargs)
            if p in TOKENIZERS:
                self.processors.append(get_tokenizer(p).from_config(p_config))
            elif p in NORMALIZERS:
                self.processors.append(get_normalizer(p).from_config(p_config))
            else:
                raise ValueError(f'Unknown tokenizer or normalizer: {p}.')


    def __call__(self, X: str, return_list: bool = False) -> t.Union[str, t.List[str]]:
        """ Run all the (pre)processors (tokenizer, normalizers,..) in a sequence.

        Args:
            X (str): a sentence (title, name, description, ...)
            return_list (bool): return list of tokens if True, space separated tokens in string otherwise

        Returns:
            (str or list[str]) of processed tokens

        """
        result = X
        for p in self.processors[:-1]:
            result = p(result, return_list=True)
        result = self.processors[-1](result, return_list=return_list)
        return result
