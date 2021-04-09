import numpy as np
from itertools import groupby
from operator import itemgetter
from typing import List


def find_ngram(x: np.ndarray, occurence: int= 2) -> List[int]: # add token
    """
    Build pairwise of size occurence of conssecutive value.
    :param x:
    :type x: ndarray
    :param occurence: number of succesive occurence
    :type occurence: int
    :return:
    :rtype:

    Example:
    --------
    >>> a = np.([1, 2, 5, 7, 8, 10])
    >>> find_ngram(a)
    [[1, 2], [5, 7], [7, 8]]
    """
    assert occurence != 0

    save_group_index= []

    for k, g in groupby(enumerate(x), lambda x: x[0] - x[1]):
        index_ngram= list(map(itemgetter(1), g))

        num_occurence= len(index_ngram)

        if num_occurence < occurence:
            continue

        elif num_occurence == occurence:
            save_group_index.append(index_ngram)

        elif (num_occurence % occurence == 0) & (num_occurence > occurence):
            generator_ngram_index= iter(index_ngram)
            group_of_ngram= [[next(generator_ngram_index) for j in range(occurence)]
                             for i in range(int(num_occurence/occurence))]
            save_group_index += group_of_ngram

        elif (num_occurence % occurence != 0) & (num_occurence > occurence):
            group_of_ngram= [[index_ngram[i+j] for j in range(occurence)]
                             for i in range(num_occurence - occurence + 1)]

            save_group_index += group_of_ngram

    return save_group_index
