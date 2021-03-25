import torch
import numpy as np


def rebuild_text(text: torch.Tensor, alphabet, space_index, sequence_len):

    if len(text.size()) == 3:
        # Delete first dim = to batch size
        text= text.squeeze(0)

    get_letter_from_index = np.vectorize(lambda index: alphabet[index])

    len_added_zero = np.argmax(np.flip(text.numpy().T.sum(axis=1)))
    fliped_matrix = np.flip(text.numpy().T[:sequence_len - len_added_zero], axis=0)

    sentence = np.where(fliped_matrix.sum(axis=1) == 0, space_index
                        , np.argmax(fliped_matrix, axis=1))

    return "".join(get_letter_from_index(sentence))