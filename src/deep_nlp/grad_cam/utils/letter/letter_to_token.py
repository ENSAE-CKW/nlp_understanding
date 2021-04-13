import re
import numpy as np
import unicodedata
from nltk.corpus import stopwords
from ..utils import implemente_multiple_time

class LetterToToken():

    def __init__(self, text: str, heatmap, pattern_token_split: str= None):

        if pattern_token_split is None:
            self._pattern_token_split= r'[\s\.\,\:\;\(\)\[\]\&\!\?\/\\]+'
        else:
            self._pattern_token_split= pattern_token_split

        self.neg_words= ["apres", "assez", "importe", "moindres", "moins", "n'", "n’", "ne", "neanmoins", "ni"
                         , "nombreuses", "nul", "sans", "sauf", "stop", "suffisant", "tellement", "pas"]

        self.text= text
        self.heatmap= heatmap
        self._index_sentence_token= self._get_index_split_pattern(self.text)
        # TODO : take off negative stopwords like "pas", "ne" etc.
        self.french_stopwords = stopwords.words('french')
        self.get_token = []
        self.get_token_clean= []
        self.get_heatmap = []
        self.get_extended_heatmap = []
        pass

    def _get_index_split_pattern(self, text):
        index_list = []
        for m in re.finditer(self._pattern_token_split, text):
            index_list.append((m.start(), m.end()))
        return index_list

    def _clean_tokens(self, token: str):
        return re.sub(self._pattern_token_split, "", token)

    def _strip_accents_and_lowercase(self, token: str, all_stop_word=False) -> str:
        if all_stop_word:
            french_stopwords = tuple([self._clean_tokens(word) for word in stopwords.words('french')
                                      if word not in self.neg_words])
        else:
            french_stopwords = tuple([])
        lower_cleaned_token = self._clean_tokens(token)
        if lower_cleaned_token not in french_stopwords:
            return lower_cleaned_token
        else:
            return ""

    @staticmethod
    def _clean_heatmap(cleaned_token, heatmap_value):
        if cleaned_token == "":
            return 0
        else:
            return heatmap_value

    # @tailrec kind
    def _compute_heatmap_pooling(self, heatmap, type="mean"):

        if type not in ["mean", "median", "max", "min", "maxabs", "minabs", "sum", "tanh", "logit"]:
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
        elif type == "sum":
            return np.sum(heatmap)
        elif type == "tanh":
            return np.tanh(self._compute_heatmap_pooling(heatmap, type="sum"))
        elif type == "logit":
            return 1/(1+np.exp(self._compute_heatmap_pooling(heatmap, type="sum")))
        else:
            raise RuntimeError

    def transform_letter_to_token(self, type= "mean"):

        step_index = 0
        for i in self._index_sentence_token:

            # Append tokens and token separators
            self.get_token.append(self.text[step_index:i[1]])
            self.get_token_clean.append(
                self._strip_accents_and_lowercase(self.text[step_index:i[1]])
            )
            value = self._clean_heatmap(
                cleaned_token= self.get_token_clean[-1]
                , heatmap_value= self._compute_heatmap_pooling(self.heatmap[step_index:i[1]], type=type)
            )
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
                self.get_token_clean.append(
                    self._strip_accents_and_lowercase(self.text[step_index:])
                )
                # value = self._compute_heatmap_pooling(self.heatmap[step_index:], type=type)
                value = self._clean_heatmap(
                    cleaned_token= self.get_token_clean[-1]
                    , heatmap_value= self._compute_heatmap_pooling(self.heatmap[step_index:], type=type)
                )
                self.get_heatmap.append(value)

                self.get_extended_heatmap = implemente_multiple_time(
                    base_list= self.get_extended_heatmap
                    , value=value
                    , times=len(self.text[step_index:])
                )

        return {"tokens": self.get_token
            , "cleaned_tokens": self.get_token_clean
            , "heatmap": np.array(self.get_heatmap)
            , "heatmap_extended": np.array(self.get_extended_heatmap)
                }
