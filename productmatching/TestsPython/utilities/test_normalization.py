import pytest
import sys

sys.path.append("/app/SourcesPython")

from utilities.normalize.normalizers import normalize_string


def test_whitespaces():
    norm = normalize_string("Produkt1  nabidka1")
    assert norm == "produkt1 nabidka1"


def test_curly_brackets_cena():
    norm = normalize_string("{{POZOR podezrela cena, ID ....}} produkt1 nazev")
    assert norm == "produkt1 nazev"


def test_curly_brackets_random():
    norm = normalize_string("Produkt1 {{Neco nahodneho}} nazev")
    assert norm == "produkt1 nazev"


def test_remove_unnecessary_symbols():
    norm = normalize_string("Pro|d;u[k@t#1 o<b>s~a@h?u}j{i=ci zn;;aky:")
    assert norm == "produkt1 obsahujici znaky"


def test_match_dates():
    norm = normalize_string("Datum 1.1.100 neboli 1/1000 rovnez 1.1000 bude odebrano")
    assert norm == "datum neboli rovnez bude odebrano"


def test_match_prices():
    norm = normalize_string("20czk neni malo, ani 100 czk (neboli 100kc), 100 eur")
    assert norm == "neni malo ani neboli"


def test_match_units():
    norm = normalize_string("Rozměry 3KG (3000 g), 1.5m (150cm), 1 ks, 2 kusy, 3 TBL")
    assert norm == "rozmery 3kg 3000g 1.5m 150cm 1ks 2kusy 3tbl"

def test_word_splits():
    norm = normalize_string("Chunks for pudl znacky a1kgb XSH8W BO635.5W")
    assert norm == "chunks pudl znacky a1kgb xsh8w bo635.5w"

def test_match_x_dni():
    norm = normalize_string("Doba 30 dní, 30 dnů")
    assert norm == "doba"


def test_match_sleva():
    norm = normalize_string("Slevy: sleva 20% neboli akce 20%")
    assert norm == "slevy neboli"


def test_concat_dims():
    norm = normalize_string("Rozmery: 100.3dm x 20 neboli 20  +100 rovnez 100cm *  30tbl x 1ks, 0 x 3")
    assert norm == "rozmery 100.3dmx20 neboli 20+100 rovnez 100cm*30tblx1ks 0x3"


def test_split_units_long():
    norm = normalize_string("produkt2kg*2cm+2tblnabidka15kg")
    assert norm == "produkt 2kg*2cm+2tbl nabidka 15kg"


def test_split_units_short():
    norm = normalize_string("produkt2kg*2cm+2tblnabidka15g")
    assert norm == "produkt 2kg*2cm+2tbl nabidka15g"
