import pandas as pd
import numpy as np
import collections
import re
import torch
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from torch.utils.data import Dataset

from typing import Dict



class sentenceToTensor(Dataset):
    def __init__(self, train_csv_path, data_csv_path, valid_csv_path, bag_of_words: Dict[str, int]= None, max_bow= None):

        self.train_csv_path= train_csv_path
        self.data_csv_path= data_csv_path
        self.valid_csv_path = valid_csv_path
        # Number mawf char we consider
        self.max_bow= max_bow

        # Alphabet definition
        self.fr_stopwords = set(stopwords.words('french'))
        add_to_stopwords= ['', 'a']
        for i in add_to_stopwords:
            self.fr_stopwords.add(i)

        # # Get french tokenizer
        # self.fr_tokenizer = lambda sentence: [x for x in sentence.split(" ") if x.lower() not in self.fr_stopwords]

        # Get french stemmer
        self.fr_stemmer = SnowballStemmer("french", ignore_stopwords= True)
        # Get only letters and ' char
        self.banned_char= '[^A-Za-z0-9\']+'
        # To delete [a-z]+'
        # self.letter_before_ap= lambda string: re.sub(r"^.*?'", '', string)

        # Load data, label, and vocab
        self.load()
        if bag_of_words is None:
            self.count_list_into_list()
        else:
            self.bag_of_words= bag_of_words

        if self.max_bow is not None:
            self.bag_of_words= dict(self.bag_of_words.most_common(self.max_bow))
    pass

    def __getitem__(self, index):
        x= self.make_bow_vector(index)
        y= self.y[index]
        return x, y


    def __len__(self):
        return len(self.label)

    def len_vocab(self):
        return len(self.bag_of_words)

    def fr_tokenizer(self, sentence):
        returned= []
        for x in sentence.split(" "):
            if x.lower() not in self.fr_stopwords:
                returned.append(x)
        return returned

    def letter_before_ap(self, string):
        return re.sub(r"^.*?'", '', string)

    def bow(self):
        return self.bag_of_words

    def clean_sentence(self, sentence):
        return [i for i in
                [self.fr_stemmer.stem(
            self.letter_before_ap(
                re.sub(self.banned_char, '', x)
            )
        ) for x in self.fr_tokenizer(sentence)]
                if i not in self.fr_stopwords]

    def load(self):
        data= pd.read_csv(self.data_csv_path)
        self.data = np.array(data["review"])
        self.label= np.array(data["label"])
        self.y= torch.LongTensor(self.label)

        # Build vocab
        data= pd.read_csv(self.train_csv_path)
        self.train= np.array(data["review"])
        data = pd.read_csv(self.valid_csv_path)
        self.valid = np.array(data["review"])

        self.all_sentence = np.concatenate((self.train, self.valid)).tolist()

        # List of list into array
        self.all_cleaned_sentence = np.array(list(map(self.clean_sentence, self.all_sentence)))
        pass

    def count_list_into_list(self):
        self.bag_of_words = collections.Counter(self.all_cleaned_sentence[0])
        for li in self.all_cleaned_sentence[1:]:
            self.bag_of_words.update(li)
        pass

    def make_bow_vector(self, index):
        # From https://pytorch.org/tutorials/beginner/nlp/deep_learning_tutorial.html
        tensor = torch.zeros(len(self.bag_of_words))
        for word in self.data[index]:
            try:
                ok= self.bag_of_words[word] # if it doesn't generate an error
                tensor[list(self.bag_of_words.keys()).index(word)] += 1
            except:
                pass

        return tensor

