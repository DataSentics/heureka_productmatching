
PREPROCESS_CONFIG = {
    'processors': [
        'originalpminormalizer',
    ],
    'configs': {
        'whitespacetokenizer': {},
        'originalpminormalizer': {
            'input_pmi': '/app/data/pmi.txt',
            'unit_conversions': True,
        },
        'donothingnormalizer': {}
    }
}
