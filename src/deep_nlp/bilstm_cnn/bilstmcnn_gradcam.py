import torch.nn as nn
import torch
from ..grad_cam.model import GradCamBaseModel


class BilstmCnn(GradCamBaseModel):
    def __init__(self,  embedding_matrix, sentence_size, input_dim, hidden_dim, layer_dim, output_dim, feature_size
                 , kernel_size=3, dropout_rate=0.5):
        super(BilstmCnn, self).__init__()

        self.linear_dim = (hidden_dim - 1) * feature_size
        self.embedding = nn.Embedding.from_pretrained(nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
                                                      , padding_idx=0)
        self.LSTM = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, bidirectional=True,
                            num_layers=layer_dim, dropout=dropout_rate, bias=True)

        # Conv Layer definition and integration
        self.convLayer = nn.Sequential(
            nn.Conv1d(in_channels=sentence_size, out_channels=feature_size, kernel_size=kernel_size, bias=True),
            nn.BatchNorm1d(feature_size),
            nn.ReLU())

        self.before_conv.add_module("conv", self.convLayer)

        # Maxpool layer definition and integration
        self.maxpool = nn.MaxPool1d(kernel_size=2)

        self.pool.add_module("maxpool", self.maxpool)

        # Classifier definition and integration
        self.fc = nn.Linear(self.linear_dim, output_dim)

        self.after_conv.add_module("fc", self.fc)

    def get_activations(self, x):
        out = self.embedding(x)
        out = torch.transpose(out, dim0=1, dim1=0)
        out, (_, _) = self.LSTM(out)
        out = torch.transpose(out, dim0=1, dim1=0)
        print(out.size())
        out= self.before_conv(out)
        return out

    def forward(self, x):
        out= self.get_activations(x)
        print(out.size())
        out = self.pool(out)
        out = out.reshape(out.size(0), -1)

        out = self.after_conv(out)

        return out