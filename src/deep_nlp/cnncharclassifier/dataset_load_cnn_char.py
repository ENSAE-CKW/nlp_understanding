import pandas as pd
import numpy as np
import string
import torch
from torch.utils.data import Dataset


class charToTensor(Dataset):
    def __init__(self, data_csv_path, sentence_max_size
                 , alphabet=None):

        self.data_csv_path= data_csv_path
        self.sentence_max_size= sentence_max_size

        # Alphabet definition
        self.all_letters= string.ascii_letters + ".,;:'/?!()@&=#0123456789\"éèêà€$"
        if alphabet is not None:
            self.all_letters= alphabet

        self.n_letters= len(self.all_letters)

        self.load() # Load data and label

    def __getitem__(self, index):
        x= self.line_to_Tensor(index)
        y= self.y[index]
        return x, y

    def __len__(self):
        return len(self.label)

    def load(self):
        data= pd.read_csv(self.data_csv_path)
        self.data= np.array(data["review"])
        self.label= np.array(data["label"])
        self.y= torch.LongTensor(self.label)
        pass

    def get_alphabet(self):
        return self.all_letters

    # From our letter vocabulary, find the letter position into all_letters list
    def letter_to_Index(self, letter):
        return self.all_letters.find(letter)

    # Turn a word/sentence into one hot tensor (size_of_sentence x 1 x n_letters)
    def line_to_Tensor(self, index):
        tensor= torch.zeros(self.n_letters, self.sentence_max_size)
        sentence= self.data[index]
        for li, letter in enumerate(sentence[:self.sentence_max_size][::-1]):
            if self.letter_to_Index(letter) != -1:  # If letter in the vocabulary list, return one hot encoded tensor
                # If letter not in the vocabulary list, return full zero vector
                tensor[self.letter_to_Index(letter)][li]= 1
        return tensor