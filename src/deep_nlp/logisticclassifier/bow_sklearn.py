import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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


def bow(train, valid, data, max_features, vocabulary= None):
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


# def train(params):
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
X_train_clean = clean_array_sentence(X_train)[:80000]
X_valid_clean = clean_array_sentence(X_valid)
print("Clean End")

######################################################
train_encoded = bow(train=X_train_clean
                          , valid=X_valid_clean
                          , max_features=params["max_features"]
                          , data=X_train_clean
                      )
vectorizer = train_encoded["vectorizer"]
vocabulary = vectorizer.vocabulary_
vocabulary_sorted= {k: v for k, v in sorted(vocabulary.items(), key=lambda item: item[1], reverse= True)}
n= 100
n_first_vocabulary= list(vocabulary_sorted.items())[:n]

X_valid_encoded = bow(train= X_train_clean
                          , valid= X_valid_clean
                          , max_features=params["max_features"]
                          , data= X_valid_clean
                          , vocabulary=n_first_vocabulary
                          )
######################################################

# Select the best model between each new_feature possibility
for i in [100, 500, 1000]:
    params["max_features"]= i
    print(params["max_features"])

    train_encoded = bow(train=X_train_clean
                          , valid=X_valid_clean
                          , params=params
                          , data=X_train_clean
                          )
    vectorizer = train_encoded["vectorizer"]
    vocabulary = vectorizer.vocabulary_

    X_train_encoded= train_encoded["data"]

    print(X_train_encoded.shape)
    print("Start Scale")
    scaler = StandardScaler()
    scaler.fit(X_train_encoded)
    print("End Scale")

    # Model training step
    # Model definition
    model = LogisticRegression(solver='saga', penalty= "l1"
                               , n_jobs= -1)

    grid_params = {
        "C": np.logspace(-3, 3, 7)
    }

    gs = GridSearchCV(model
                      , param_grid=grid_params
                      , scoring="neg_log_loss"
                      , verbose= 1)

    print("Model training phase")
    X_train_encoded_scale= scaler.transform(X_train_encoded)
    del X_train_encoded
    gs.fit(X_train_encoded_scale, y_train)
    del X_train_encoded_scale

    X_valid_encoded = bow(train=X_train_clean
                          , valid=X_valid_clean
                          , params=params
                          , data=X_valid_clean
                          , vocabulary=vocabulary
                          )["data"]

    print("Validation phase")
    X_valid_encoded_scale = scaler.transform(X_valid_encoded)
    del X_valid_encoded
    y_pred = gs.predict(X_valid_encoded_scale)
    print("Max_feature : {} | AUC : {:.6f}".format(i, roc_auc_score(y_valid, y_pred)))
    del X_valid_encoded_scale

    continue

    # pass



if __name__ == "__main__":

    params = {
        "model_path": r"data/06_models/logisticclassifier/allocine_classification"
        , "train_path": r"data/01_raw/allocine_train.csv"
        , "valid_path": r"data/01_raw/allocine_valid.csv"
        , "model_saved_name": "/model_allocine.pth.tar"
        , "log_file_name": "/train_log.txt"
        , "max_features": 500
    }

    # train= pd.read_csv(params["train_path"])
    # X= np.array(train["review"])[:10]
    # print(clean_text(X[0]))
    train(params)