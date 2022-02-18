import dataclasses
import numpy as np
import typing as t

from utilities.priceoutlier.utils import Distribution
from utilities.priceoutlier.bartlett import bartlett
from utilities.priceoutlier.constant import constant


@dataclasses.dataclass
class OutlierTests:
    constant: Distribution
    bartlett: float


@dataclasses.dataclass
class PriceOutlierResult:
    distribution: Distribution
    tests: t.Optional[OutlierTests]


def priceoutlier(
    value: float,
    data: t.List[float],
    const: float,
    significance: float,
) -> PriceOutlierResult:
    # Note that we treat Unknown test results (due to insufficient data) as Standard data points.
    # This was implemented via PURPLE-2344 due to imbalance in nr. of offers for candidates, which
    #   resulted in the model basing its decision on this (misleading) metric.
    #   (see also PURPLE-2397)
    if not data:
        return PriceOutlierResult(
            distribution=Distribution.unknown,
            tests=OutlierTests(
                constant=Distribution.unknown,
                bartlett=1.0,
            )
        )

    sorted_data = np.sort(np.array(data))
    is_higher = value > sorted_data[-1]
    is_lower = value < sorted_data[0]

    if not is_higher and not is_lower:
        return PriceOutlierResult(
            distribution=Distribution.standard,
            tests=OutlierTests(
                constant=Distribution.standard,
                bartlett=1.0,
            )
        )

    constant_result = constant(
        value=value,
        sorted_data=list(sorted_data),
        c=const,
    )
    bartlett_pval, bartlett_result = bartlett(
        value=value,
        data=data,
        significance=significance,
    )

    is_outlier = constant_result == Distribution.outlier or bartlett_result == Distribution.outlier

    return PriceOutlierResult(
        distribution=Distribution.outlier if is_outlier else Distribution.standard,
        tests=OutlierTests(
            constant=constant_result,
            bartlett=bartlett_pval,
        )
    )
