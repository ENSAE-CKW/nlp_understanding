import torch
import torch.nn as nn
from typing import Iterable, TypeVar, Any

class BERTCNNClassifier(nn.Module):
    # Merge between  deep_nlp/embed_cnn/embcnnmodel (by khaled)
    # and https://github.com/alisafaya/OffensEval2020/blob/master/models.py
    def __init__(self, bert
                 , max_sentence_size: int, embedding_dim: int, nb_filter: int, height_filter: Iterable[int]
                 , output_dim: int, dropout: float):

        super(BERTCNNClassifier, self).__init__()

        # Get our BERT model
        self.bert= bert

        # Construct our CNN
        self.conv = nn.ModuleList()

        for height in height_filter:
            conv_lay = nn.Sequential(
                nn.Conv2d(1, int(nb_filter), (int(float(height)), embedding_dim)),
                nn.ReLU(),
                nn.MaxPool2d((max_sentence_size - height + 1, 1), stride=1),
            )
            self.conv.append(conv_lay)

        self.fc = nn.Linear(len(height_filter) * nb_filter, output_dim)
        self.log_softmax = nn.LogSoftmax(dim= 1)
        self.dp = nn.Dropout(p= dropout)
        pass

    def forward(self, x, mask, token_type_ids):
        x= self.bert(x, attention_mask= mask, token_type_ids= token_type_ids)[2][-4:] # why -4: ?
        x = torch.stack(x, dim= 1)
        x = [conv(x).squeeze() for conv in self.conv]
        x = torch.cat(tuple(x), dim=1) #flaten into [nb_batch x sum_nb_filters]
        x = self.dp(x) # doesn't change the shape of x
        x = self.fc(x)  # [nb_batch, 2]
        x = self.log_softmax(x) # Better than Softmax (convergence)
        return x