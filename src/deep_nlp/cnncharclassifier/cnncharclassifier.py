import torch.nn as nn

class CNNCharClassifier(nn.Module):

    def __init__(self, params):
        # Paste from https://github.com/srviest/char-cnn-text-classification-pytorch/blob/master/model.py
        super(CNNCharClassifier, self).__init__() # legacy
        self.params= params
        self.seq_len= self.params["cnn_seq_len"]
        self.feature_num= self.params["cnn_feature_num"] # vocab size
        self.feature_size= self.params["cnn_feature_size"]
        self.kernel_one= self.params["cnn_kernel_one"] # 7
        self.kernel_two= self.params["cnn_kernel_two"] # 3
        self.stride_one= self.params["cnn_stride_one"] # 1
        self.stride_two= self.params["cnn_stride_two"] # 3
        self.input_linear= int(((self.seq_len - 96)/27)*self.feature_size)
        self.output_linear= self.params["cnn_output_linear"]
        self.num_class= int(self.params["cnn_num_class"]) # 2

        self.conv1 = nn.Sequential(
            nn.Conv1d(self.feature_num, self.feature_size, kernel_size= self.kernel_one, stride= self.stride_one),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size= self.kernel_two, stride= self.stride_two)
        )

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

        self.conv6 = nn.Sequential(
            nn.Conv1d(self.feature_size, self.feature_size, kernel_size= self.kernel_two, stride= self.stride_one),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size= self.kernel_two, stride= self.stride_two)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(self.input_linear, self.output_linear),
            nn.ReLU(),
            nn.Dropout(p=self.params["cnn_dropout"])
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.output_linear, self.output_linear),
            nn.ReLU(),
            nn.Dropout(p=self.params["cnn_dropout"])
        )

        self.fc3 = nn.Linear(self.output_linear, self.num_class)
        self.log_softmax = nn.LogSoftmax(dim= 1)

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    nn.init.normal_(m, 0, 0.05)
            except:
                pass

    def forward(self, x):
        x= self.conv1(x) # size [feature_num x feature_size] (== [vocabulary_size x feature_num])
        x= self.conv2(x) # size [feature_size x feature_size]
        x= self.conv3(x) # size [feature_size x feature_size]
        x= self.conv4(x) # size [feature_size x feature_size]
        x= self.conv5(x) # size [feature_size x feature_size]
        x= self.conv6(x) # size [feature_size x feature_size]

        # collapse
        x= x.view(x.size(0), -1)
        # linear layer
        x= self.fc1(x) # size input: input_size | size output: output_size
        # linear layer
        x= self.fc2(x) # size input: output_size | size output: output_size
        # linear layer
        x= self.fc3(x) # size input: output_size | size output: num_class
        # output layer
        x= self.log_softmax(x)
        return x