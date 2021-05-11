import torch.nn as nn
import torch
import torch.nn.functional as F
from ..grad_cam.model import GradCamBaseModel

class classifier3F(GradCamBaseModel):
    # define all the layers used in model
    def __init__(self, wv, no_words, embedding_dim, nb_filter, height_filter, output_dim, dropout, padded):
        # Constructor
        super().__init__()
        self.height_filter = height_filter
        self.padded = padded
        self.no_words= no_words
        # embedding layer
        self.embedding = nn.Embedding.from_pretrained(wv)

        self.conv1_conv= nn.ModuleList()
        for height in height_filter:
            self.conv1_conv.append(
                nn.Conv2d(in_channels=1, out_channels=int(nb_filter), kernel_size=(int(float(height)), embedding_dim))
            )
            self.before_conv.add_module("conv1_conv_{}".format(height), self.conv1_conv[-1])


        self.conv1_relu = nn.ReLU()
        self.before_conv.add_module("conv1_relu", self.conv1_relu)

        if self.padded:
            self.conv1_maxpool= nn.Sequential(
                nn.MaxPool1d(no_words, stride=1)
            )
            self.pool.add_module("conv1_maxpool", self.conv1_maxpool)
        else:
            self.conv1_maxpool= nn.ModuleList()
            for height in height_filter:
                self.conv1_maxpool.append(
                    nn.MaxPool1d(no_words - height + 1, stride=1)
                    # nn.MaxPool1d(no_words, stride=1)
                )
                self.pool.add_module("conv1_maxpool_{}".format(height), self.conv1_maxpool[-1])


        self.fc = nn.Linear(len(height_filter) * nb_filter, output_dim)

        self.sm = nn.LogSoftmax(dim=1)

        self.dp = nn.Dropout(p=dropout)

        self.after_conv.add_module("dp", self.dp)
        self.after_conv.add_module("fc", self.fc)
        self.after_conv.add_module("sm", self.sm)

        self.params = [wv, no_words, embedding_dim, nb_filter, height_filter, output_dim, dropout, padded]

    def get_params(self):
        return self.params

    def get_activations(self, x):
        # Documentation said to !!!
        # Each forward step, reset gradient list to only get the one from the actual run (=from this forward step)
        self.reset_gradient_list()

        x = self.embedding(x)
        x = x.unsqueeze(1)

        conv_before= []
        for i in range(len(self.height_filter)):
            conv_before.append(self.before_conv[i])
        relu = self.before_conv.conv1_relu

        if self.padded:
            x_padded = [nn.ZeroPad2d((0, 0, 0, height - 1))(x) for height in self.height_filter]
            x_padded = list(zip(x_padded, conv_before))
            x = [relu(conv(x)).squeeze(3) for x, conv in x_padded]
        else:
            x = [relu(conv(x)).squeeze(3)for conv in conv_before]
        return x

    def forward(self, x):
        # Documentation said to !!!
        x = self.get_activations(x)

        # Documentation said to !!!
        # Apply relu after convolution layer

        for i in x:
            # Documentation said to !!!
            if i.requires_grad:
                h= self.register_hook(i)

        if self.padded:
            x_copy= [self.pool(i).squeeze(2) for i in x]
        else:
            x_copy= []
            for i in range(len(self.height_filter)):
                x_copy.append(self.pool[i](x[i]).squeeze(2))

        x = torch.cat(tuple(x_copy), dim=1)

        x= self.after_conv(x)
        return x
