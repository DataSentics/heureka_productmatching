import sys

sys.path.append("/app/SourcesPython")

from model_extras.unit_checker import UnitChecker
from utilities.normalize.normalizers import normalize_string

def wrapper_uc_results(a, b, res, conversions=False):
    unit_checker = UnitChecker(unit_conversions=conversions, verbose=False)
    a_norm = normalize_string(a, unit_conversions=conversions)
    b_norm = normalize_string(b, unit_conversions=conversions)
    uc_res = unit_checker(a_norm, b_norm)
    assert uc_res.a_units == res['a'] and uc_res.b_units == res['b']

def wrapper_uc_decision(a, b, decision, conversions=False):
    unit_checker = UnitChecker(unit_conversions=conversions, verbose=False)
    a_norm = normalize_string(a, unit_conversions=conversions)
    b_norm = normalize_string(b, unit_conversions=conversions)
    uc_res = unit_checker(a_norm, b_norm)
    assert uc_res.decision == decision

def test_basic_usage():
    a = 'Vanish 5 l'
    b = 'Mňamka 200 g'
    res = {'a': {'l': {((), '5')}},
           'b': {'g': {((), '200')}}}
    wrapper_uc_results(a, b, res)

def test_multiplication():
    a = '22x Miamor Ragout Royale losos 100g'
    b = 'Jed 100*2ml'
    res = {'a': {'g': {(('22',), '100')}},
           'b': {'ml': {(('100',), '2')}}}
    wrapper_uc_results(a, b, res)

def test_addition_without_units():
    a = '2 + 5x100cm'
    b = '4x10x10mb'
    res = {'a': {'cm': {(('2', '5'), '100')}},
           'b': {'mb': {(('10', '4'), '10')}}}
    wrapper_uc_results(a, b, res)

def test_addition_multiple_units():
    a = 'Pytel cementu 10kg + 2x100g bonus'
    b = 'Brambory rané 2x10kg + 3 + 2x200g'
    res = {'a': {'kg': {((), '10')}, 'g': {(('2',), '100')}},
           'b': {'kg': {(('2',), '10')}, 'g': {(('2', '3'), '200')}}}
    wrapper_uc_results(a, b, res)

def test_stock_units_multiplication():
    a = 'Sodastream Ladies Edition náhradní láhve (500ml/2ks)'
    b = 'Sodastream láhev Fuse Sport Games bílá modrá červená 3 ks 1 l'
    res = {'a': {'ml': {(('2',), '500')}},
           'b': {'l': {(('3',), '1')}}}
    wrapper_uc_results(a, b, res)

def test_stock_units_vanilla():
    a = '2ks'
    b = '5kusu'
    res = {'a': {'ks': {((), '2')}},
           'b': {'ks': {((), '5')}}}
    wrapper_uc_results(a, b, res)

def test_stock_units_tricky():
    a = 'Marp Holistic víčko na konzervy 800g 1ks'
    b = '2+500g 3ks'
    res = {'a': {'g': {((), '800')}},
           'b': {'g': {(('2', '3'), '500')}}}
    wrapper_uc_results(a, b, res)

def test_conversion():
    a = '1dm'
    b = '1.1kg'
    res = {'a': {'m': {((), '0.1')}},
           'b': {'g': {((), '1100')}}}
    wrapper_uc_results(a, b, res, True)

def test_conversion_case():
    a = '50 dB'
    b = '1 Ml'
    res = {'a': {'db': {((), '50')}},
           'b': {'ml': {((), '1')}}}
    wrapper_uc_results(a, b, res, True)

def test_conversion_multiple_units():
    a = '20 m 100 kg'
    b = '2*20 dm 3*100 kg'
    res = {'a': {'m': {((), '20')}, 'g': {((), '100000')}},
           'b': {'m': {(('2',), '2')}, 'g': {(('3',), '100000')}}}
    wrapper_uc_results(a, b, res, True)

def test_positive_decision():
    a = 'Limča 2x500ml'
    b = 'Limonáda 0.5l 2ks'
    wrapper_uc_decision(a, b, 'ok', True)

def test_decision_without_units():
    a = 'Vanish'
    b = 'Vanish 5l'
    wrapper_uc_decision(a, b, 'ok', True)

def test_decision_different_values():
    a = '.51 GB'
    b = '500 MB'
    wrapper_uc_decision(a, b, 'no', True)

def test_decision_different_units():
    a = 'Párátko 20cm'
    b = 'Sada párátek 100kg'
    wrapper_uc_decision(a, b, 'ok', True)
