import numpy as np
from typing import List
from math import ceil


def split_into_batches(
    data: List,
    batch_size: int = 0,
    n_batches: int = None,
) -> List:
    """
    Generator for splitting `data` into batches.
    When `n_batches` is provided, the `batch_size` is ignored.
    """
    if n_batches:
        batch_size = ceil(len(data) / min(len(data), n_batches))

    for batch_first_ind in range(0, len(data), batch_size):
        batch = data[batch_first_ind:(batch_first_ind + batch_size)]
        yield batch


def argmax_full(array: List) -> List:
    """
    Finds all occurences of maximal value in a given iterable (not generator), O(n).
    """
    if len(array) == 0:
        return []

    return np.where(np.max(array) == array)[0]
