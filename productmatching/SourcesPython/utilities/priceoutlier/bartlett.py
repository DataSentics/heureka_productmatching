import numpy as np
import scipy.stats as ss
import typing as t

from utilities.priceoutlier.utils import Distribution

# hide warnings, caused by zero variance in data
np.seterr(divide='ignore')

def bartlett(
    value: float,
    data: t.List[float],
    significance: float = 0.05,
) -> t.Tuple[float, Distribution]:
    if len(data) < 2:
        return 1.0, Distribution.unknown

    group_1, group_2 = np.array(data), np.array(data + [value])
    pval = ss.bartlett(group_1, group_2).pvalue
    if pval < significance:
        return pval, Distribution.outlier

    return pval, Distribution.standard
