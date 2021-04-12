import torch
import torch.nn as nn
from abc import abstractmethod
from ..grad_cam.model import GradCamBaseModel


class CNNCharClassifier(GradCamBaseModel):

    def __init__(self, sequence_len, feature_num, feature_size, kernel_one, kernel_two, stride_one, stride_two
                 , output_linear, num_class, dropout):

        super(CNNCharClassifier, self).__init__()
        self.sequence_len= sequence_len #+ 6#9222 #sequence_len
        self.feature_num= feature_num # vocab size
        self.feature_size= feature_size
        self.kernel_one= kernel_one # 7
        self.kernel_two= kernel_two # 3
        self.stride_one= stride_one # 1
        self.stride_two= stride_two # 3
        self.input_linear= int(((self.sequence_len - 96)/27)*self.feature_size)
        self.output_linear= output_linear
        self.num_class= int(num_class) # 2
        self.dropout= dropout

        # paddng argument
        # self.to_pad= nn.ZeroPad2d((0, 9222 - 1014, 0, 0))
        # self.to_pad = nn.ZeroPad2d((0, (1014 + 6) - 1014, 0, 0))


        self.conv1_conv= nn.Conv1d(self.feature_num, self.feature_size, kernel_size= self.kernel_one, stride= self.stride_one)
        self.conv1_relu= nn.ReLU()
        self.conv1_maxpool= nn.MaxPool1d(kernel_size= self.kernel_two, stride= self.stride_two)


        self.conv2 = nn.Sequential(
            nn.Conv1d(self.feature_size, self.feature_size, kernel_size= self.kernel_one, stride= self.stride_one),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size= self.kernel_two, stride= self.stride_two)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(self.feature_size, self.feature_size, kernel_size= self.kernel_two, stride= self.stride_one),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(self.feature_size, self.feature_size, kernel_size= self.kernel_two, stride= self.stride_one),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(self.feature_size, self.feature_size, kernel_size= self.kernel_two, stride= self.stride_one),
            nn.ReLU()
        )

        self.conv6= nn.Sequential(
            nn.Conv1d(self.feature_size, self.feature_size
                                    , kernel_size=self.kernel_two, stride=self.stride_one)
            , nn.ReLU()
            , nn.MaxPool1d(kernel_size=self.kernel_two, stride=self.stride_two)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(self.input_linear, self.output_linear),
            nn.ReLU(),
            nn.Dropout(p=self.dropout)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.output_linear, self.output_linear),
            nn.ReLU(),
            nn.Dropout(p=self.dropout)
        )

        self.fc3 = nn.Linear(self.output_linear, self.num_class)
        self.log_softmax = nn.LogSoftmax(dim= 1)

        # Init weight
        self.weight_init()

        # Access before maxpooling
        self.before_conv.add_module("conv1_conv", self.conv1_conv)
        self.before_conv.add_module("conv1_relu", self.conv1_relu)

        # disect the network to access its last convolutional layer
        self.pool.add_module("conv1_maxpool", self.conv1_maxpool)

        # get the max pool of the features stem
        self.after_conv.add_module("conv2", self.conv2)
        self.after_conv.add_module("conv3", self.conv3)
        self.after_conv.add_module("conv4", self.conv4)
        self.after_conv.add_module("conv5", self.conv5)
        self.after_conv.add_module("conv6", self.conv6)

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    nn.init.normal_(m, 0, 0.05)
            except:
                pass

    def forward(self, x):
        x = self.get_activations(x)

        if x.requires_grad:
            h= self.register_hook(x)

        x = self.pool(x)

        x = self.after_conv(x)

        x = x.view(x.size(0), -1)
        # Can't incorporate FC part into after_conv because of this x.view between conv and FC
        x= self.fc1(x)
        x= self.fc2(x)
        x= self.fc3(x)
        x= self.log_softmax(x)
        return x