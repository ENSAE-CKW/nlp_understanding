import torch.nn as nn
import torch
from deep_nlp.grad_cam.model import GradCamBaseModel


class BilstmCnn(GradCamBaseModel):
    def __init__(self,  embedding_matrix, sentence_size, input_dim, hidden_dim, layer_dim, output_dim, feature_size
                 , kernel_size, dropout_rate, padded):
        super(BilstmCnn, self).__init__()

        self.padded = padded
        self.kernel_size= kernel_size
        self.mp_kernel_size = 2

        if not self.padded:
            self.linear_dim = int(2*hidden_dim * (sentence_size-kernel_size)/2)
        else:
            self.linear_dim = int(2 * hidden_dim * (sentence_size - 1)/ 2)

        self.embedding = nn.Embedding.from_pretrained(nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
                                                      , padding_idx=0)
        self.LSTM = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, bidirectional=True,
                            num_layers=layer_dim, dropout=dropout_rate, bias=True)

        self.convLayer = nn.Sequential(
            nn.Conv1d(in_channels=2*hidden_dim, out_channels=feature_size, kernel_size=kernel_size, bias=True),
            nn.BatchNorm1d(2*hidden_dim),
            nn.ReLU())

        self.maxpool = nn.MaxPool1d(kernel_size=self.mp_kernel_size)

        self.fc = nn.Linear(self.linear_dim, output_dim)

        self.softmax = nn.Softmax(dim=1)

        # Fill up pipelines
        self.before_conv.add_module("conv", self.convLayer)
        self.pool.add_module("maxpool", self.maxpool)
        self.after_conv.add_module("fc", self.fc)
        self.after_conv.add_module("sm", self.softmax)

    def get_activations(self, x):
        # Documentation said to !!!
        # Each forward step, reset gradient list to only get the one from the actual run (=from this forward step)
        self.reset_gradient_list()

        if self.padded:
            x= nn.ZeroPad2d((0, self.kernel_size - 1, 0, 0))(x)

        x = self.embedding(x)
        x= x.permute(1, 0, 2)
        x, (_, _) = self.LSTM(x)
        x = x.permute(1, 2, 0)
        x= self.before_conv(x)
        return x

    def forward(self, x):
        x= self.get_activations(x)

        if x.requires_grad:
            h= self.register_hook(x)

        x= self.pool(x)

        x = x.reshape(x.size(0), -1)

        x= self.after_conv(x)

        return x