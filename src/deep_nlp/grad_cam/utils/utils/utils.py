import numpy as np

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