import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from deep_nlp.cnncharclassifier import CNNCharClassifier, charToTensor
import pickle

from deep_nlp.grad_cam.utils.letter import rebuild_text, prepare_heatmap, LetterToToken
from deep_nlp.grad_cam.plot import plot_bar_heatmap, plot_text_and_heatmap


import matplotlib.pyplot as plt
import itertools
import re

cnn_sequence_len= 1014
cnn_feature_num= 87
cnn_feature_size= 256
cnn_kernel_one= 7
cnn_kernel_two= 3
cnn_stride_one= 1
cnn_stride_two= 3
cnn_output_linear= 1024
cnn_num_class= 2
cnn_dropout= 0.5
cnn_cuda_allow= True

model_path_saved= r"data/06_models/cnn_char_classifier/cnn_char_model/cnn_char_model.pt"

with open(model_path_saved, 'rb') as f:
    model_saved= pickle.load(f)

data_df= pd.read_csv(r"data/01_raw/allocine_test.csv")

test_data= charToTensor(data_df= data_df, sentence_max_size= cnn_sequence_len)

test_load = DataLoader(test_data, batch_size= 1
                       , num_workers=4
                       , drop_last=True, shuffle=True)

test_sentence= next(iter(test_load))

# Initialisation
parameters = {"sequence_len": cnn_sequence_len, "feature_num": cnn_feature_num
    , "feature_size": cnn_feature_size, "kernel_one": cnn_kernel_one
    , "kernel_two": cnn_kernel_two, "stride_one": cnn_stride_one
    , "stride_two": cnn_stride_two, "output_linear": cnn_output_linear
    , "num_class": cnn_num_class, "dropout": cnn_dropout}


model = CNNCharClassifier(**parameters)

if cnn_cuda_allow:
    model = torch.nn.DataParallel(model).cuda()
else:
    model = torch.nn.DataParallel(model)

model.load_state_dict(model_saved)

state_dict= model.module.module.state_dict() # delete module to allow cpu loading

cpu_model= CNNCharClassifier(**parameters).cpu()
cpu_model.load_state_dict(state_dict)

cpu_model.eval()

output= cpu_model(test_sentence[0])

# Compute heatmap from basemodel object
heatmap_test= cpu_model.get_heatmap(text= test_sentence[0]
                                    , num_class= 1
                                    , dim= [0, 2]
                                    , type_map= "normalized")
heatmap_test= heatmap_test[0]

# CNNChar rebuilt from input the text
alphabet= test_data.get_alphabet()+" "
rebuild_sentence= rebuild_text(text= test_sentence[0]
                                 , alphabet= alphabet
                                 , space_index= len(alphabet) - 1 #83 # ajout de +4 si pas fait
                                 , sequence_len= cnn_sequence_len)

# Resize heatmap Brutal method
heatmap_match_sentence_size_invert= prepare_heatmap(heatmap= heatmap_test
                                                    , text= rebuild_sentence)

# Plot character level
plot_bar_heatmap(heatmap_match_sentence_size_invert)

plot_text_and_heatmap(text= rebuild_sentence
                      , heatmap= heatmap_match_sentence_size_invert
                      , figsize=(7, 7)
                      , cmap= "RdYlGn")

# Plot token level
## Transform character level to token one
letter_to_token= LetterToToken(text= rebuild_sentence
                               , heatmap= heatmap_match_sentence_size_invert)

results_dict= letter_to_token.transform_letter_to_token(type= "tanh")
tokens= results_dict["tokens"]
heatmap_test= results_dict["heatmap"]

plot_bar_heatmap(heatmap_test)

plot_text_and_heatmap(text= tokens
                      , heatmap= heatmap_test
                      , figsize=(7, 7)
                      , cmap= "PiYG"
                      , word_or_letter= "word")

