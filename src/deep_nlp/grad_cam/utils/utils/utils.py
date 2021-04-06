import numpy as np
import itertools
from typing import List, Dict, Any
import torch

# Match heatmap size and text size
def resize_array(array_d, target_size):
    # target size is the text length
    assert len(array_d.shape) == 1 # 1D-array only (List like)

    array_size= array_d.shape[0]

    if target_size <= array_size:
        return array_d[:target_size]
    else:
        return np.pad(array_d, (0, target_size - array_size))
    pass


def implemente_multiple_time(base_list, value, times):
    return base_list + [value] * times


def transform_to_list_freq(x):
    return list(itertools.chain.from_iterable(x))


def preprocess_before_barplot(results: List[Any]):
    mots_plus_75 = [element[0]["mots_expli"] for element in results if element[0]["prob"] > 0.75]
    mots_50_75 = [element[0]["mots_expli"] for element in results if
                  (element[0]["prob"] > 0.5) & (element[0]["prob"] < 0.75)]
    mots_25_50 = [element[0]["mots_expli"] for element in results if
                  (element[0]["prob"] > 0.25) & (element[0]["prob"] < 0.5)]
    mots_0_25 = [element[0]["mots_expli"] for element in results if (element[0]["prob"] < 0.25)]

    mots_plus_75 = transform_to_list_freq(mots_plus_75)
    mots_50_75 = transform_to_list_freq(mots_50_75)
    mots_25_50 = transform_to_list_freq(mots_25_50)
    mots_0_25 = transform_to_list_freq(mots_0_25)

    return mots_plus_75, mots_50_75, mots_25_50, mots_0_25