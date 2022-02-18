from .normalizers import normalize_string, BaseNormalizer, DoNothingNormalizer, OriginalPmiNormalizer
from typing import Union

NORMALIZERS = {
    'originalpminormalizer': OriginalPmiNormalizer,
    'donothingnormalizer': DoNothingNormalizer,
    '': DoNothingNormalizer,  # valid, no normalization
}

def get_normalizer(normalizer: Union[str, BaseNormalizer]):
    if isinstance(normalizer, BaseNormalizer):
        return normalizer
    else:
        try:
            return NORMALIZERS[normalizer.strip()]
        except KeyError:
            raise ValueError(f'Normalizer ({normalizer}) does not exist. Should be one of {", ".join(NORMALIZERS.keys())}.')
