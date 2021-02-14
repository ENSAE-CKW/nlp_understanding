# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List

import numpy as np
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

# For Logistic Bag Of Word

def clean_text(sentence: str) -> List[str]:

    # Remove non-letter characters
    sentence= re.sub("[^a-zA-Z]", " ", sentence).lower().split()
    fr_stemmer = SnowballStemmer("french")
    fr_stopwords = set(stopwords.words('french'))

    cleaned_sentence= [fr_stemmer.stem(word) for word in sentence
                       if word not in fr_stopwords]

    return (" ".join(cleaned_sentence))


def bow(train, data, max_features, vocabulary= None):
    if vocabulary is None:
        vectorizer= CountVectorizer(analyzer = "word"
                                    , tokenizer = None
                                    , preprocessor = None
                                    , stop_words = None
                                    , max_features= max_features
                                    , dtype=np.uint8
                                    )
        # Construct our vocabulary
        vectorizer.fit(train)

    else:
        vectorizer = CountVectorizer(analyzer = "word"
                                    , tokenizer = None
                                    , preprocessor = None
                                    , stop_words = None
                                     , max_features= max_features
                                     , vocabulary= vocabulary
                                     , dtype= np.uint8
                                     )

    data = vectorizer.transform(data).toarray().astype('uint8')

    return {"vectorizer": vectorizer, "data": data}

# Clean our review
def clean_array_sentence(array: np.ndarray) -> List[List[str]]:
    return list(map(clean_text, array))


def encode_data(data: pd.DataFrame, max_number_feature: int
                , vocabulary= None):

    # Split X and Y data
    if vocabulary is None:
        print("train")
        X_train = np.array(data["review"])
        y_train = np.array(data["label"])

        print("Clean Start")
        X_train_clean = clean_array_sentence(X_train)  # needed to construct vocabulary
        print("Clean End")

        train_encoded = bow(train=X_train_clean
                            , max_features=max_number_feature
                            , data=X_train_clean
                            )
        vectorizer = train_encoded["vectorizer"]
        vocabulary = vectorizer.vocabulary_

        print(train_encoded["data"].shape)
        return [train_encoded["data"], y_train, vocabulary]

    else:
        print("other")
        X_data = np.array(data["review"])
        y_data = np.array(data["label"])

        # Clean our sentence
        print("Clean Start")
        X_data_clean = clean_array_sentence(X_data)
        print("Clean End")


        # Fake input (I know this is not a good way to code ... but this is the way)
        X_data_encoded = bow(train= np.array([0])
                                , max_features= max_number_feature
                                , data= X_data_clean
                              , vocabulary=vocabulary
                              )["data"]
        print(X_data_encoded.shape)
        return [X_data_encoded, y_data]
    pass