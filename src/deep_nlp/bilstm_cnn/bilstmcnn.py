import torch.nn as nn
import torch

class BilstmCnn(nn.Module):
    def __init__(self,  embedding_matrix, sentence_size, input_dim, hidden_dim, layer_dim, output_dim, feature_size, kernel_size=3, dropout_rate=0.5):
        super(BilstmCnn, self).__init__()

        self.mp_kernel_size = 2
        self.linear_dim = int(2*hidden_dim * (sentence_size-kernel_size)/2)
        self.embedding = nn.Embedding.from_pretrained(nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32)), padding_idx=0)
        self.LSTM = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, bidirectional=True,
                            num_layers=layer_dim, dropout=dropout_rate, bias=True)

        self.convLayer = nn.Sequential(
            nn.Conv1d(in_channels=2*hidden_dim, out_channels=feature_size, kernel_size=kernel_size, bias=True),
            nn.BatchNorm1d(2*hidden_dim),
            nn.ReLU())

        self.maxpool = nn.MaxPool1d(kernel_size=self.mp_kernel_size)

        self.fc = nn.Linear(self.linear_dim, output_dim)

    def forward(self, x):
        out = self.embedding(x)

        out = out.permute(1,0,2)

        out, (hn, cn) = self.LSTM(out)

        out = out.permute(1,2,0)

        out = self.convLayer(out)

        out = self.maxpool(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc(out)

        return out