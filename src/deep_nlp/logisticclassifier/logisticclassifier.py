import torch.nn as nn


class LogisticClassifier(nn.Module):
    def __init__(self, params):
        super(LogisticClassifier, self).__init__()

        self.params= params
        self.feature_num = int(self.params["feature_num"]) # vocab size
        self.num_class= int(self.params["num_class"])

        # Architecture
        self.linear= nn.Linear(self.feature_num, self.num_class)
        self.log_softmax= nn.LogSoftmax(dim= 1)
        pass

    def forward(self, x):
        x= x.view(x.size(0), -1)
        x= self.linear(x)
        x= self.log_softmax(x)
        return x