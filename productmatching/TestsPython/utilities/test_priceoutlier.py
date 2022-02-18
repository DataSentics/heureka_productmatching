import sys

sys.path.append("/app/SourcesPython")
from utilities.priceoutlier.priceoutlier import priceoutlier
from utilities.priceoutlier.bartlett import bartlett
from utilities.priceoutlier.constant import constant
from utilities.priceoutlier.utils import Distribution


def test_bartlett_unknown():
    pv, dist = bartlett(
        value=1.0,
        data=[2.0],
    )
    assert pv == 1.0


def test_constant_unknown():
    assert constant(value=1.0, sorted_data=[], c=1.0) == Distribution.unknown


def test_constant_standard():
    data = sorted([1.0, 2.0, 3.0, 4.0])

    in_range = constant(value=2.5, sorted_data=data, c=1.0)
    assert in_range == Distribution.standard

    below = constant(value=0.5, sorted_data=data, c=2.0)
    assert below == Distribution.standard

    above = constant(value=8, sorted_data=data, c=2.0)
    assert above == Distribution.standard


def test_constant_outlier():
    data = sorted([1.0, 2.0, 3.0, 4.0])

    below = constant(value=0.4, sorted_data=data, c=2.0)
    assert below == Distribution.outlier

    above = constant(value=9.0, sorted_data=data, c=2.0)
    assert above == Distribution.outlier


def test_priceoutlier():
    data = [10.2, 14.1, 14.4, 14.4, 14.4, 14.5, 14.5, 14.6, 14.7]
    assert priceoutlier(
        value=13.0,
        data=data,
        const=2.0,
        significance=0.05,
    ).distribution == Distribution.standard
