import typing as t

from utilities.priceoutlier.utils import Distribution


def constant(
    value: float,
    sorted_data: t.List[float],
    c: float = 2.0,
) -> Distribution:
    if len(sorted_data) < 1:
        return Distribution.unknown

    elif value > sorted_data[-1] and value / sorted_data[-1] > c:
        return Distribution.outlier

    elif value < sorted_data[0] and sorted_data[0] / value > c:
        return Distribution.outlier

    else:
        return Distribution.standard
