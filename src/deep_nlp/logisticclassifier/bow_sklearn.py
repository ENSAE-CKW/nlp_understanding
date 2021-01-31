import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from typing import List

import matplotlib.pyplot as plt


def clean_text(sentence: str) -> List[str]:

    # Remove non-letter characters
    sentence= re.sub("[^a-zA-Z]", " ", sentence).lower().split()
    # fr_tokenizer = RegexpTokenizer(r'''\w'|\w+|[^\w\s]''')
    fr_stemmer = SnowballStemmer("french")
    fr_stopwords = set(stopwords.words('french'))

    # cleaned_sentence= [fr_stemmer.stem(token.lower())
    #                    for token in fr_tokenizer.tokenize(sentence)
    #                    if token.lower() not in fr_stopwords]
    cleaned_sentence= [fr_stemmer.stem(word) for word in sentence
                       if word not in fr_stopwords]

    return (" ".join(cleaned_sentence))


def bow(train, valid, data, params, vocabulary= None):
    if vocabulary is None:
        vectorizer= CountVectorizer(analyzer = "word"
                                    , tokenizer = None
                                    , preprocessor = None
                                    , stop_words = None
                                    , max_features= params["max_features"]
                                    , dtype=np.uint8
                                    )
        # Construct our vocabulary
        vectorizer.fit(train)

    else:
        vectorizer = CountVectorizer(analyzer = "word"
                                    , tokenizer = None
                                    , preprocessor = None
                                    , stop_words = None
                                     , max_features= params["max_features"]
                                     , vocabulary= vocabulary
                                     , dtype= np.uint8
                                     )

    data = vectorizer.transform(data).toarray().astype('uint8')

    return {"vectorizer": vectorizer, "data": data}


def train(params):
    train = pd.read_csv(params["train_path"])
    X_train = np.array(train["review"])
    y_train = np.array(train["label"])
    del train

    valid = pd.read_csv(params["valid_path"])
    X_valid = np.array(valid["review"])
    y_valid = np.array(valid["label"])
    del valid

    # Clean our review
    def clean_array_sentence(array: np.ndarray) -> List[List[str]]:
        return list(map(clean_text, array))

    print("Clean Start")
    X_train_clean = clean_array_sentence(X_train)
    X_valid_clean = clean_array_sentence(X_valid)
    print("Clean End")

    print("Encode Start")
    train_encoded = bow(train=X_train_clean
                        , valid=X_valid_clean
                        , params=params
                        , data=X_train_clean
                        )
    print("Encode End")

    # Construct our base vocabulary with max features
    # We are going to select the number of features in
    # the next for bound
    vectorizer = train_encoded["vectorizer"]
    vocabulary = vectorizer.vocabulary_
    del train_encoded

    # Select the best model between each new_feature possibility
    for i in [100]:
        params["max_features"]= i
        print(params["max_features"])

        X_train_encoded = bow(train=X_train_clean
                              , valid=X_valid_clean
                              , params=params
                              , data=X_train_clean
                              , vocabulary=vocabulary
                              )["data"]


        # Model training step
        # Model definition
        model = LogisticRegression(solver='liblinear')

        grid_params = {
            "C": np.logspace(-3, 3, 7)
            , "penalty": ["l1"]
        }

        gs = GridSearchCV(model
                          , param_grid=grid_params
                          , scoring="neg_log_loss")

        print("Model training phase")
        gs.fit(X_train_encoded, y_train)

        del X_train_encoded

        X_valid_encoded = bow(train=X_train_clean
                              , valid=X_valid_clean
                              , params=params
                              , data=X_valid_clean
                              , vocabulary=vocabulary
                              )["data"]

        print("Validation phase")
        y_pred = gs.predict(X_valid_encoded)
        print("Max_feature : {} | AUC : {:.6f}".format(i, roc_auc_score(y_valid, y_pred)))
        del X_valid_encoded

        continue

    pass



if __name__ == "__main__":

    params = {
        "model_path": r"../../../data/06_models/logisticclassifier/allocine_classification"
        , "train_path": r"../../../data/01_raw/allocine_train.csv"
        , "valid_path": r"../../../data/01_raw/allocine_valid.csv"
        , "model_saved_name": "/model_allocine.pth.tar"
        , "log_file_name": "/train_log.txt"
        , "max_features": 10000
    }

    # train= pd.read_csv(params["train_path"])
    # X= np.array(train["review"])[:10]
    # print(clean_text(X[0]))
    train(params)