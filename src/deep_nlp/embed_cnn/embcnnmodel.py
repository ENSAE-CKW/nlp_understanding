import torch.nn as nn
import torch

class classifier3F(nn.Module):
    # TODO : remove embedding_dim using wv.shape[1]
    # define all the layers used in model
    def __init__(self, wv, no_words, embedding_dim, nb_filter, height_filter, output_dim, dropout):
        # Constructor
        super().__init__()

        # embedding layer
        self.embedding = nn.Embedding.from_pretrained(wv)

        # Ne pas oublier d'ajouter un view !
        # Convolutionnal layer
        # it uses initialization as proposed by Kaiming et.al

        self.conv = nn.ModuleList()

        for height in height_filter:
            conv_lay = nn.Sequential(
                nn.Conv2d(1, int(nb_filter), (int(float(height)), embedding_dim)),
                nn.ReLU(),
                nn.MaxPool2d((no_words - height + 1, 1), stride=1),
            )
            self.conv.append(conv_lay)

        self.fc = nn.Linear(len(height_filter) * nb_filter, output_dim)

        self.sm = nn.Softmax(dim=1)

        self.dp = nn.Dropout(p=dropout)

        self.params = [wv, no_words, embedding_dim, nb_filter, height_filter, output_dim, dropout]

    def get_params(self):
        return self.params

    def forward(self, text):
        x = self.embedding(text)
        x = x.unsqueeze(1)  # [nb_batch, nb_channel = 1, nb_words_in_sentences, embedding_dim]
        x = [conv(x).squeeze() for conv in self.conv]
        x = torch.cat(tuple(x), dim=1) #flaten into [nb_batch x sum_nb_filters]
        x = self.dp(x) #doesn't change the shape of x
        x = self.fc(x)  # [nb_batch, 2]
        x = self.sm(x)
        return x