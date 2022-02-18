import re
import logging
import typing as t
from collections import namedtuple, defaultdict

from utilities.normalize.regexes import get_units, conversions

NumUnitCheck = namedtuple("NumUnitCheck", "decision a_units b_units")


class UnitChecker():
    def __init__(self, unit_conversions: bool = True, verbose: bool = True):
        if verbose:
            logging.debug(f"Creating unitchecker, unit conversions: {unit_conversions}")
        self.unit_conversions = unit_conversions

        units = get_units(unit_conversions)
        self.NUM = units["NUM"]
        self.NUM_UNIT = units["NUM_UNIT"]
        self.UNIT = units["UNIT"]
        self.SIGN = units["SIGN"]
        self.UNIT_SHORT = units["UNIT_SHORT"]
        self.UNIT_PREFIXES = units["UNIT_PREFIXES"]
        self.UNIT_STOCK = units["UNIT_STOCK"]
        self.NUM_STOCK_UNIT = f"({units['NUM']})({units['UNIT_STOCK']})"
        self.NUM_UNIT_OR_STOCK_UNIT = f"(({self.NUM_UNIT})|({self.NUM_STOCK_UNIT}))"
        # used only if unit conversions will be done
        self.convertable_units = [f"{pref.lower()}{un}" for un in self.UNIT_SHORT.split("|") for pref in self.UNIT_PREFIXES.split("|")+[""]]
        # all considered unit prefixes in long form (tera, kilo, ...) in regex form
        self.prefixes_to_conversions_r = "|".join(list(conversions.keys()))

    @staticmethod
    def round_integer(number: float):
        """ Auxiliary function, converts 5.0 -> 5 but 5.1 -> 5.1"""

        if number.is_integer():
            return round(number)
        else:
            return number

    def to_base_unit(self, unit: str, values: t.Set[tuple] = None):
        """Convert values for given unit to base unit values"""

        # assuming nontrivial unit consists of prefix and basic unit
        r = re.compile(
            rf"({self.prefixes_to_conversions_r})({self.UNIT_SHORT})"
        )
        found = re.search(r, unit)
        if found:
            # get prefix and base unit
            prefix, base_unit = found.group(1, 2)
            if not values:
                return base_unit, None
            # convert value according to conversiong mapping
            new_values = set(
                (mult, str(self.round_integer(float(val)*conversions[prefix])))
                for mult, val in values
            )
            return base_unit, new_values
        else:
            return unit, values

    def convert_units(self, unit_to_values: dict):
        """Converts all units to base units and adjust value"""

        # assuming input structure {"g": ((2, 3), "450")} or {"milig": ((), "450")}
        converted_units = defaultdict(set)
        for unit, values in unit_to_values.items():
            conv_unit, conv_values = self.to_base_unit(unit, values)
            converted_units[conv_unit] |= conv_values

        return converted_units

    def squash_blocks(self, s):
        # merge logical blocks
        # '100g+2x jehne 100g' -> '100g+2x100g'
        # '800g box 3 x kapsicky x 85 g raznici' -> '800g box 3x85g raznici'
        # '800g box 3 x kapsicky x 85 graznici' -> '800g box 3x85 graznici'
        rs = rf"({self.NUM})\s*(({self.UNIT})(\W)*)"
        r = re.compile(
            # from: num \s* (unit \W+)? \s* (multiplicator sign, spaces and words)|(spaces and words, multiplicator sign) \s* num \s* (unit \W+)?
            # to: num unit? multiplicator sign num unit?
            rf"({rs}?)\s*(((x|\*)(\s+([a-z]+\s+)+))|((\s+([a-z]+\s+)+)(x|\*)))\s*({rs})?"
        )

        return re.sub(r, r"\2\4\8\14\16\17", s)

    def extract_blocks(self, s):
        # find blocks like '2+2x100g'
        r = re.compile(
            r"(%s)((%s)(%s)|( (%s)))*\b"
            % (self.NUM_UNIT_OR_STOCK_UNIT, self.SIGN, self.NUM_UNIT, self.NUM_UNIT_OR_STOCK_UNIT)
        )
        # blocks like '2+2x100g' or '12345'
        init_res = [rf.group(0).strip() for rf in re.finditer(r, s)]
        # there must be at least one unit in a block to avoid pure numbers
        res = [bl for bl in init_res if re.search(r"%s" % self.UNIT, bl)]
        return sorted(res)

    def process_sub_block(self, sb: str) -> tuple:
        def parse_numunit(numunit: str) -> tuple:
            num, unit = re.sub(rf"({self.NUM})({self.UNIT})", r"\1 \2", numunit).split(' ')
            return num, unit

        # take block like "2+2x10g" or "2x2x10g"
        # return ("g", ("10", ("2", "2")))
        # split on x * or + but not if surrounded by letters from both sides, e.g. do not split "megapixel"
        spl = re.split(r"(?<![a-z])[x\+\*]|[x\+\*](?![a-z])", sb)

        if len(spl) == 1:
            # simple unit like '85g'
            num, unit = parse_numunit(spl[0])
            # exclude cases like 003 which may raise from something like xyz003mm
            if num.startswith("0.") or not num.startswith("0"):
                return (unit, (tuple(), num))
        else:
            for i, s in enumerate(spl):
                if re.search(self.UNIT, s):
                    multipliers = tuple(sorted(spl[:i] + spl[i + 1:]))
                    num, unit = parse_numunit(s)
                    if num.startswith("0.") or not num.startswith("0"):
                        return (unit, (multipliers, num))

    def split_one_block(self, s):
        # splits block '2x100g+3+2x200g' to ['2x100g', '3+2x200g']
        # we assume that first "+" after a block like "2+3+1x2x3x123gx666" separates it from another such block

        # extract info about stock unit and change it to multiplication
        # ex.: 1lx3(ks|kusy|...) -> 3x1l, 1kgx1(ks|kusy|...) -> 1kg, 2kusy -> 2ks
        r = re.compile(
            rf"({self.NUM})({self.UNIT_STOCK})"
        )
        nr_stock_units = re.search(r, s)
        if nr_stock_units is not None:
            s_wo_stock = re.sub(r, '', s).replace(' ', '')
            n_stock, units = nr_stock_units.groups()
            if not s_wo_stock:  # there are only stock units
                s_stock_multi = s.replace(units, 'ks')
            elif int(n_stock) > 1:  # multiple pieces of the same product
                s_stock_multi = f'{nr_stock_units.group(1)}*{s_wo_stock}'
            else:  # only one piece of the product
                s_stock_multi = s_wo_stock
            s = s_stock_multi

        r = re.compile(
            rf"({self.NUM}(\+|x|\*))*({self.NUM})({self.UNIT})(($|\s|\+)|((x|\*){self.NUM})+)"
        )
        num_atr_results = []
        for m in re.finditer(r, s):
            sub_block = m.group().strip(" +x*")
            num_atr_results.append(self.process_sub_block(sub_block))
        num_atr_results = [nar for nar in num_atr_results if nar is not None]

        num_atr_dict = defaultdict(set)
        # merge the dicts
        for nar in num_atr_results:
            # nar = ("g", ((2, 3), 450))
            num_atr_dict[nar[0]].add(nar[1])

        return num_atr_dict

    def process_string(self, s):
        squashed = self.squash_blocks(s)
        blocks = self.extract_blocks(squashed)
        splitted = [self.split_one_block(b) for b in blocks]
        # [{"g": ((2, 3), 450)}, {"kw": ((,), 200)}]
        merged_splitted = defaultdict(set)
        for dic in splitted:
            for k, v in dic.items():
                merged_splitted[k] |= v

        return merged_splitted.copy()

    @staticmethod
    def enrich_processed(processed, attributes):
        # processed ~= {'l': {((), '2'), ((), '1')}}
        # attributes ~= {'g': [10, 20]}
        attrs_processed = {
            unit: {((), value) for value in values}
            for unit, values in attributes.items()
        }
        # merge
        for unit, set_values in attrs_processed.items():
            processed['unit'] = processed.get('unit') | set_values
        return processed

    def __call__(self, a: str, b: str, a_attributes: dict = None, b_attributes: dict = None):
        a_processed = self.process_string(a)
        if a_attributes:
            a_processed = self.enrich_processed(a_processed, a_attributes)

        b_processed = self.process_string(b)
        if b_attributes:
            b_processed = self.enrich_processed(b_processed, b_attributes)

        if self.unit_conversions:
            # convert to base unit
            a_processed = self.convert_units(a_processed)
            b_processed = self.convert_units(b_processed)

        # define units to compare, take all common
        comparable_units = set(a_processed.keys()) & set(b_processed.keys())

        decision = 'ok'
        for cu in comparable_units:
            # compute for each unit/ck
            a_units, b_units = a_processed.get(cu, []), b_processed.get(cu, []),
            n_b_extra = sum([li not in a_units for li in b_units])
            n_a_extra = sum([li not in b_units for li in a_units])

            if n_a_extra == 0 and n_b_extra == 0:
                pass
            elif n_a_extra > 0 and n_b_extra > 0:  # possibly change this
                decision = 'no'
                break
            else:
                decision = 'unknown'
                break

        return NumUnitCheck(decision, a_processed, b_processed)
