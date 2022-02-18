import re

unit_prefixes = {
    "T": "tera",
    "G": "giga",
    "M": "mega",
    "k": "kilo",
    "h": "hecto",
    "d": "deci",
    "dc": "deci",
    "c": "centi",
    "m": "mili",
}

# multipliers wrt base unit (g, l, ...)
conversions = {
    "tera": 1_000_000_000_000,
    "giga": 1_000_000_000,
    "mega": 1_000_000,
    "kilo": 1_000,
    "hecto": 100,
    "deci": 0.1,
    "centi": 0.01,
    "mili": 0.001,
}

# not all prefixes are meaningful with every unit, define combinations to find
allowed_prefixes = {
    "b": ["M", "G", "T"],
    "w": ["k", "M"],
    "m": ["m", "c", "d", "dc", "k"],
    "m2": ["m", "c", "d", "dc", "k"],
    "m3": ["m", "c", "d", "dc", "k"],
    "l": ["h", "c", "dc", "d", "m"],
    "g": ["m", "k"],
    "hz": ["k", "M", "G"],
}


def get_units(unit_conversions=False):

    SIGN = r"x|\+|\*"
    SIGN0 = r"x\+\*"
    NUM = r"(?:[.,]\d+|\d+(?:[.,]\d*)?)"
    UNIT_SLASH = r"ot./min|ot/min|ot./s|l/h|l/min|l/s|l/100km|km/h|m/s|g/h|g/m2|g/min|m3/hod.|m3/hod|m3/h|m3/min|cd/m2|kb/s|mb/s|gb/s|kwh/rok|kwh/24h"

    UNIT0 = r"wattu|watt|tbl|tablet|rpm|pixelu|pix|palcu|ohmu|ohm|mpx|mpix|mililitry|mililitru|mililitr|megapixelu|megapixel|mah"
    UNIT1 = r"lm|litry|litru|litr|lb|inch|hz|hodin|hod|db|bitu|bit|balenie|baleni|bal|ah|\%"
    UNIT2 = r"mw|mm|ml|mhz|kw|kg|dm|dl|dkg|dg|dcl|gb|mb|cm3|cm2|cm|ghz|khz|kwh|kbps|kb"
    UNIT_STOCK = r"kusy|kusu|kus|ks"
    UNIT_SHORT = r"b|w|m|l|g|m2|m3"
    UNIT_LONG = rf"{UNIT_SLASH}|{UNIT0}|{UNIT1}|{UNIT2}|{UNIT_STOCK}"

    UNIT_PREFIXES = r"|".join(unit_prefixes.keys())
    UNIT_SHORT_CASE = None

    # add nontrivial units (prefix+base)
    if unit_conversions:
        # case sensitive units needed for conversion regex
        UNIT_SHORT_CASE = r"B|W|m|l|g|m3|m2|Hz"
        UNIT_SHORT = "|".join(set(un.lower() for un in UNIT_SHORT_CASE.split("|")))
        # after prefix replacement, work with text in lowercase again
        # choose only allowed combinations
        UNIT_NONTRIV = "|".join(set(f'{unit_prefixes[all_pref]}{un}' for un in UNIT_SHORT.split("|") for all_pref in allowed_prefixes[un]))
        UNIT_LONG = rf"{UNIT_SLASH}|{UNIT0}|{UNIT1}|{UNIT2}|{UNIT_STOCK}|{UNIT_NONTRIV}"

    UNIT = rf"{UNIT_LONG}|{UNIT_SHORT}"
    NUM_UNIT = fr"{NUM}({UNIT})?"
    CURRENCY = r"kc|czk|eur"

    return {
        "NUM": NUM, "UNIT": UNIT, "NUM_UNIT": NUM_UNIT, "UNIT_SHORT": UNIT_SHORT,
        "UNIT_SHORT_CASE": UNIT_SHORT_CASE, "UNIT_LONG": UNIT_LONG, "UNIT_STOCK": UNIT_STOCK,
        "UNIT_PREFIXES": UNIT_PREFIXES, "SIGN": SIGN, "SIGN0": SIGN0, "CURRENCY": CURRENCY
    }


def get_regexes(unit_conversions=False):

    units = get_units(unit_conversions)

    NUM = units["NUM"]
    UNIT = units["UNIT"]
    NUM_UNIT = units["NUM_UNIT"]
    UNIT_SHORT_CASE = units["UNIT_SHORT_CASE"]
    UNIT_LONG = units["UNIT_LONG"]
    SIGN = units["SIGN"]
    SIGN0 = units["SIGN0"]
    CURRENCY = units["CURRENCY"]
    UNIT_PREFIXES = units["UNIT_PREFIXES"]

    CONVERT_REGEX = None
    if unit_conversions:
        # convert prefixes to long form, e.g. kg -> kilog, ml -> milil, Hz - > Hz
        # only allowed combinations
        # match "5 kg" -> "5 kilog" but "5g" -> "5g" (base unit) and "5Mg" -> "5Mg" (not allowed combination)
        CONVERT_REGEX = (
            re.compile(r"([\d\s\(])(%s)(%s)(%s|$|\s|,|\)|/)" % (UNIT_PREFIXES, UNIT_SHORT_CASE, SIGN)),
            lambda x: f"{x.group(1)}{unit_prefixes[x.group(2)] if x.group(2) in allowed_prefixes[x.group(3).lower()] else x.group(2)}{x.group(3)}{x.group(4)}"
        )

    REGEXES = [
        # Normalize whitespaces
        (re.compile(r"\s+"), " "),
        # Match anything in {{ }} or {}, e.g. {{POZOR, podezřelá cena, ...}}
        (re.compile(r"{{(.*)}}\s*"), ""),
        (re.compile(r"{(.*)}\s*"), ""),
        # Remove unnecessary symbols
        (re.compile(r"\'|\"|„|<|>|®|°|~|“|\^|@|#|!|\?|:|\(|\)|\||;|\[|\]|{|}|=|÷", re.IGNORECASE), ""),
        (re.compile(r"[.,](\s|$)"), r"\1"),
        # Match various dates
        (re.compile(r"(exp\.*\s*)?\d{1,2}(\/|\.)\d{1,2}(\/|\.)\d{2,4}"), ""),
        (re.compile(r"(exp\.*\s*)\d{1,2}(\/|\.)\d{2,4}"), ""),
        (re.compile(r"\d{1,2}(\/|\.)\d{4}"), ""),
        # Match prices
        (re.compile(r"%s\s*(\.-)??(%s)\b" % (NUM, CURRENCY), re.IGNORECASE), ""),
        # 50 kg -> 50kg etc.
        (re.compile(r"(%s)\s*(%s)\b" % (NUM, UNIT)), r"\1\2"),
        # 5.0kg -> 5kg, 5.20kg -> 5.2kg
        (re.compile(r"(\d+)\.0(%s)\b" % (UNIT)), r"\1\2"),
        (re.compile(r"(\d+\.\d*(?<!0))0+(%s)\b" % (UNIT)), r"\1\2"),
        # Match "30 dni""
        (re.compile(r"%s\s*dn.?" % (NUM)), ""),
        # Match "sleva 20%""
        (re.compile(r"(akc|slev).?\s*%s\s*\%%" % (NUM)), ""),
        # Match "20% sleva"
        (re.compile(r"%s\s*\%%\s*(slev|akc).?" % (NUM)), ""),
        # 100 x 100 -> 100x100, etc.
        (
            re.compile(
                r"(^|\W)(%s)\s*(%s)\s*(%s)\s*(%s)\s*(%s)\s*(%s)\s*(%s)"
                % (NUM_UNIT, SIGN, NUM_UNIT, SIGN, NUM_UNIT, SIGN, NUM_UNIT)
            ),
            r"\1\2\4\5\7\8\10\11"
        ),
        (
            re.compile(
                r"(^|\W)(%s)\s*(%s)\s*(%s)\s*(%s)\s*(%s)"
                % (NUM_UNIT, SIGN, NUM_UNIT, SIGN, NUM_UNIT)
            ),
            r"\1\2\4\5\7\8"
        ),
        (
            re.compile(
                r"(^|\W)(%s)\s*(%s)\s*(%s)"
                % (NUM_UNIT, SIGN, NUM_UNIT)
                ),
            r"\1\2\4\5"
        ),
        # "abc 3 x sth" -> "abc 3x sth"
        (
            re.compile(
                r"(^|\s)(%s)\s*(%s)($|\s)"
                % (NUM_UNIT, SIGN)
                ),
            r"\1\2\4\5"
        ),
        # "2kg abc + 3kg" -> "2kg abc +3kg"
        (
            re.compile(
                r"(^|\s)(%s)\s*(%s)($|\s)"
                % (SIGN, NUM_UNIT)
                ),
            r"\1\2\3\5"
        ),
        # "jehne2gx100+200mlx300tbl-hovezi" -> "jehne 2gx100+200mlx300tbl -hovezi"
        # the last num-unit block must contain unit in this case
        (
            re.compile(
                r"(^|[^.%s\d])(%s)(%s)(%s)(%s)(%s)(%s)((%s)(%s))([^%s]|$)"
                % (SIGN0, NUM_UNIT, SIGN, NUM_UNIT, SIGN, NUM_UNIT, SIGN, NUM, UNIT, SIGN0)
            ),
            r"\1 \2\4\5\7\8\10\11 \14"
        ),
        (
            re.compile(
                r"(^|[^.%s\d])(%s)(%s)(%s)(%s)((%s)(%s))([^%s]|$)"
                % (SIGN0, NUM_UNIT, SIGN, NUM_UNIT, SIGN, NUM, UNIT, SIGN)
            ),
            r"\1 \2\4\5\7\8 \11"
        ),
        (
            re.compile(
                r"(^|[^.%s\d])(%s)(%s)((%s)(%s))([^%s]|$)"
                % (SIGN0, NUM_UNIT, SIGN, NUM, UNIT, SIGN)
            ),
            r"\1 \2\4\5 \8"
        ),
        # "a123kg" -> "a 123kg", "-123kg-" -> "- 123kg -"
        # no change: "a123", "a123kgb", "XSH8W", "BO635.5W"
        (
            re.compile(
                r"(^|[^.%s\d])((%s)(%s))([^%s\w]|$)"
                % (SIGN0, NUM, UNIT_LONG, SIGN0)
            ),
            r"\1 \2 \5"
        ),
        (
            re.compile(
                r"(^|[^.%s\w])((%s)(%s))([^%s\w]|$)"
                % (SIGN0, NUM, UNIT, SIGN0)
            ),
            r"\1 \2 \5"
        ),
        # Normalize whitespaces, again
        (re.compile(r"\s+"), " "),
    ]

    return REGEXES, CONVERT_REGEX
