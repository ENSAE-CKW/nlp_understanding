import re
import numpy as np
from ..utils import implemente_multiple_time

class LetterToToken():

    def __init__(self, text: str, heatmap, pattern_token_split: str= None):

        if pattern_token_split is None:
            self._pattern_token_split= r'[\s\.\,\:\;\(\)\[\]\&\!\?]+'
        else:
            self._pattern_token_split= pattern_token_split

        self.text= text
        self.heatmap= heatmap
        self._index_sentence_token= self._get_index_split_pattern(self.text)
        self.get_token = []
        self.get_heatmap = []
        self.get_extended_heatmap = []
        pass

    def _get_index_split_pattern(self, text):
        index_list = []
        for m in re.finditer(self._pattern_token_split, text):
            index_list.append((m.start(), m.end()))
        return index_list

    # @tailrec kind
    def _compute_heatmap_pooling(self, heatmap, type="mean"):

        if type not in ["mean", "median", "max", "min", "maxabs", "minabs"]:
            raise ValueError

        if type == "mean":
            return np.mean(heatmap)
        elif type == "median":
            return np.median(heatmap)
        elif type == "max":
            return np.max(heatmap)
        elif type == "min":
            return np.min(heatmap)
        elif type == "maxabs":
            return self._compute_heatmap_pooling(np.abs(heatmap), type="max")
        elif type == "minabs":
            return self._compute_heatmap_pooling(np.abs(heatmap), type="min")
        else:
            raise RuntimeError

    def transform_letter_to_token(self, type= "mean"):
        step_index = 0

        for i in self._index_sentence_token:
            if i[0] == 0:
                # If the first character is a token separator
                self.get_token.append(self.text[0:i[1]])
                value = self._compute_heatmap_pooling(self.heatmap[0:i[1]], type= type)
                self.get_heatmap.append(value)

                self.get_extended_heatmap = implemente_multiple_time(
                    base_list= self.get_extended_heatmap
                    , value= value
                    , times= i[1] + 1
                )

                step_index = i[1]

            # # Append tokens
            # get_token.append(text_test[step_index:i[0]])
            # get_heatmap.append(np.mean(heatmap_test[step_index:i[0]]))
            # # Append token separators
            # get_token.append(text_test[i[0]:i[1]])
            # get_heatmap.append(np.mean(heatmap_test[i[0]:i[1]]))

            # Append tokens and token separators
            self.get_token.append(self.text[step_index:i[1]])
            value = self._compute_heatmap_pooling(self.heatmap[step_index:i[1]], type=type)
            self.get_heatmap.append(value)

            self.get_extended_heatmap = implemente_multiple_time(
                base_list= self.get_extended_heatmap
                , value=value
                , times=i[1] - step_index
            )

            # Update beginning of the next token
            step_index = i[1]

            # After the last token separator, if there is a token, add it
            if (self._index_sentence_token[-1] == i) & (self.text[step_index:] != ''):
                self.get_token.append(self.text[step_index:])
                value = self._compute_heatmap_pooling(self.heatmap[step_index:], type=type)
                self.get_heatmap.append(value)

                self._get_extended_heatmap = implemente_multiple_time(
                    base_list= self._get_extended_heatmap
                    , value=value
                    , times=len(self.text[step_index:])
                )

        return {"tokens": self.get_token
            , "heatmap": np.array(self.get_heatmap)
            , "heatmap_extended": np.array(self.get_extended_heatmap)
                }
