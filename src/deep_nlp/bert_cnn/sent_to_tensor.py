import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, RandomSampler
import transformers
import unicodedata
import re


class bertToTensor(Dataset):
    def __init__(self, data_df: pd.DataFrame, max_seq_len: int
                 , tokenizer: transformers.PreTrainedTokenizerFast):

        self.data_df= data_df
        self.max_seq_len= max_seq_len
        self.tokenizer= tokenizer

        self.load() # Load data and label
        self.tokenized_data= self.tokenize()

    def __getitem__(self, index):
        seq, mask= self.tokenize_seq_to_tensor()
        x_y = TensorDataset(seq, mask, self.y) # Wrapping tensors
        return x_y

    def __len__(self):
        return len(self.label)

    def load(self):
        self.data= np.array(self.data_df["review"]) # care self.
        self.label= np.array(self.data_df["label"])
        self.y= torch.LongTensor(self.label)
        pass

    def preprocess_text(self, text):
        # From https://stackoverflow.com/a/29920015/5909675
        matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', text.replace("#", " "))
        return " ".join([m.group(0) for m in matches])

    def strip_accents_and_lowercase(self, text):
        return ''.join(c for c in unicodedata.normalize('NFD', text)
                       if unicodedata.category(c) != 'Mn').lower()

    def tokenize(self):
        # Use our BERT tokenizer to convert ["Ce film est vraiment nul"]
        # into {'input_ids': [[101, 2023, 2003, 1037, 14324, 2944, 14924, 4818, 102, 0]]*
        # , 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 0]]}
        text= [self.strip_accents_and_lowercase(self.preprocess_text(t)) for t in self.data]

        return self.tokenizer.batch_encode_plus(
            text
            , max_length= self.max_seq_len
            , pad_to_max_length= "max_length"
            , truncation= True
            , return_token_type_ids= False)

    def tokenize_seq_to_tensor(self):
        data_seq= torch.tensor(self.tokenized_data["input_ids"])
        data_mask= torch.tensor(self.tokenized_data["attention_mask"])
        return data_seq, data_mask
