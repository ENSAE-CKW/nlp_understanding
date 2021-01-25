import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer


class sentenceToTensor(Dataset):
    def __init__(self, params, data_path):
        self.params= params
        self.train_path= self.params["train_path"]
        self.valid_path = self.params["valid_path"]
        self.data_path = data_path

        self.fr_tokenizer = RegexpTokenizer(r'''\w'|\w+|[^\w\s]''')

        self.fr_stemmer = SnowballStemmer("french")

        self.fr_stopwords = set(stopwords.words('french'))

        self.load()
        pass

    def __getitem__(self, index):
        x= self.sentenceToTensor(index)
        y= self.y[index]
        return x, y

    def __len__(self):
        return len(self.label)

    def preprocessor(self, sentence):
        analyzer= CountVectorizer().build_analyzer()
        return (self.fr_stemmer.stem(x) for x in analyzer(sentence) \
                if x not in self.fr_stopwords)

    def load(self):
        # load data to transform into tensor
        data = pd.read_csv(self.data_path)
        self.data = np.array(data["review"])
        self.label = np.array(data["label"])
        self.y = torch.LongTensor(self.label)

        # Call our counter
        self.vectorizer = CountVectorizer(analyzer= self.preprocessor
                                     , tokenizer= self.fr_tokenizer
                                     , max_features= self.params["max_features"]
                                     )
        if self.params["vocabulary"] is not None:
            self.vectorizer= CountVectorizer(analyzer= self.preprocessor
                                     , tokenizer= self.fr_tokenizer
                                     , max_features= self.params["max_features"]
                                     , vocabulary= self.params["vocabulary"]
                                     )

        else:
            # load and create out vocab
            train = pd.read_csv(self.train_path)
            self.train = np.array(train["review"])

            valid = pd.read_csv(self.valid_path)
            self.valid = np.array(valid["review"])

            self.vectorizer.fit(self.train)
            self.vectorizer.fit(self.valid)

        self.data= self.vectorizer.transform(self.data).toarray()

    pass

    def get_vectorizer(self):
        return self.vectorizer

    def get_params(self, deep= True):
        return self.vectorizer.get_params(deep)

    def sentenceToTensor(self, index):
        sentence= self.data[index]
        return torch.Tensor(sentence)