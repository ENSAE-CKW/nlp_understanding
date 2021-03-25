import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from deep_nlp.cnncharclassifier import CNNCharClassifier, charToTensor
import pickle

import matplotlib.pyplot as plt
import itertools
import re

cnn_sequence_len= 1014
cnn_feature_num= 83
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

# Call our model
parameters = {"sequence_len": cnn_sequence_len, "feature_num": cnn_feature_num
    , "feature_size": cnn_feature_size, "kernel_one": cnn_kernel_one
    , "kernel_two": cnn_kernel_two, "stride_one": cnn_stride_one
    , "stride_two": cnn_stride_two, "output_linear": cnn_output_linear
    , "num_class": cnn_num_class, "dropout": cnn_dropout}

model = CNNCharClassifier(**parameters)
model = torch.nn.DataParallel(model)
if cnn_cuda_allow:
    model = torch.nn.DataParallel(model).cuda()
model.load_state_dict(model_saved)

state_dict= model.module.module.state_dict() # delete module to allow cpu loading

cpu_model= CNNCharClassifier(**parameters).cpu()
cpu_model.load_state_dict(state_dict)

cpu_model.eval()

output= cpu_model(test_sentence[0])

grd= cpu_model.activations_grad_cam(test_sentence[0], num_class= 0) # batch_size x num_features x feature_map_length
pooled_grd= torch.mean(grd, dim= [0, 2])

activations= cpu_model.get_activations(test_sentence[0]).detach()

for i in range(activations.shape[1]):
            activations[:, i, :] *= pooled_grd[i]

heatmap = torch.mean(activations, dim=1).squeeze()
heatmap = np.maximum(heatmap, 0)
heatmap /= torch.max(heatmap)
heatmap= heatmap.numpy()


# Text rebuilder from the model input
alphabet= test_data.get_alphabet()+" "
get_letter_from_index = np.vectorize(lambda index: alphabet[index])

def rebuild_text(text: torch.Tensor):
    space_index = 83

    len_added_zero = np.argmax(np.flip(text.numpy().T.sum(axis=1)))
    fliped_matrix = np.flip(text.numpy().T[:cnn_sequence_len - len_added_zero], axis=0)

    sentence = np.where(fliped_matrix.sum(axis=1) == 0, space_index
                        , np.argmax(fliped_matrix, axis=1))

    return "".join(get_letter_from_index(sentence))


# Match heatmap size and text size
def resize_array(array_d, target_size):
    # target size is the text length
    assert len(array_d.shape) == 1 # 1D-array only (List like)

    array_size= array_d.shape[0]

    if target_size <= array_size:
        return array_d[:target_size]
    else:
        return np.pad(array_d, (0, target_size - array_size))
    pass


def prepare_heatmap(heatmap, text):
    sentence_len = len(text)
    # Cut the heatmap to the sentence len (character level)
    resize_heatmap= resize_array(heatmap, sentence_len)

    # The first character is the last in the input model
    # So we need to invert our heatmap
    return resize_heatmap[::-1]


def reshape_text_to_plot(text, heatmap, threshold= 40):
    # Reshape text and heatmap to allow grid plotting
    # Threshold isthe number of character per grid row
    text_adjusted= None
    heatmap_adjusted= None

    sentence_len = len(text)
    heatmap_len = heatmap.shape[0]
    assert sentence_len == heatmap_len

    if sentence_len >= threshold:
        entire_part= sentence_len // threshold

        # easy case : 80 // 40 = 2 ==> 40*2==80 = true
        if entire_part * threshold == sentence_len:
            text_adjusted= np.array(list(rebuild_sentence)).reshape((threshold, entire_part))
            heatmap_adjusted= heatmap.reshape((threshold, entire_part))
        else:
            epsilon= threshold - (sentence_len - (entire_part * threshold))

            # Adjust size with 0 or space character
            text_adjusted = text + " " * epsilon
            heatmap_adjusted = np.pad(heatmap, (0, epsilon))

            # Then reshape
            text_adjusted = np.array(list(text_adjusted)).reshape((entire_part + 1, threshold))
            heatmap_adjusted = heatmap_adjusted.reshape((entire_part + 1, threshold))
    else:
        diff_len= threshold - sentence_len

        text_adjusted= text + " " * diff_len # add space to match threshold value
        text_adjusted= np.array(list(text_adjusted))
        heatmap_adjusted= np.pad(heatmap, (0, diff_len)) # same here but for array (add 0)

    return text_adjusted, heatmap_adjusted


def reshape_list_to_plot(text, heatmap, threshold= 10):
    num_token= len(text)
    token_adjusted= None
    heatmap_adjusted= None

    if num_token >= threshold:
        # easy case : 80 // 40 = 2 ==> 40*2==80 = true
        entire_part= num_token // threshold

        if entire_part * threshold == num_token:
            num_rows= entire_part
        else:
            num_rows= entire_part + 1

        # Now we need to define the number of "columns" needed (= the max of num character from rows)
        stock_num_character_per_row= []
        step_index= 0
        for i in range(num_rows): # TODO:  delete those awful double "for" bound
            max_token= threshold*(i+1)
            tokens_rows= text[step_index:max_token]

            token_per_row_len= 0
            for token in tokens_rows:
                try:
                    assert type(token) in [str, np.str_]
                except:
                    raise ValueError("Need a str object")
                token_per_row_len += len(token)


            # Get the num of character per row
            stock_num_character_per_row.append(token_per_row_len)

            # Update index of first token to take
            step_index= max_token

        # Get the num of col needed to create a matrix [num_rows x num_col]
        num_col= np.max(stock_num_character_per_row)

        # With num col and rows, adjust heatmap dimension and token
        heatmap_adjusted= []
        token_adjusted= []
        step_index = 0
        for i in range(num_rows):
            max_token = threshold * (i + 1)

            heatmap_rows= heatmap[step_index:max_token]
            tokens_rows = text[step_index:max_token]
            token_adjusted.append(tokens_rows)

            new_heatmap_row= []
            for j in range(len(heatmap_rows)):

                new_heatmap_row= implemente_multiple_time(new_heatmap_row
                                                          , value= heatmap_rows[j]
                                                          , times= len(tokens_rows[j]))

            # If the heatmap adjusted (by the number of token) is under the num of col, add some 0
            diff_len= num_col - len(new_heatmap_row)
            # heatmap_adjusted.append(np.pad(new_heatmap_row, (0, diff_len)))
            heatmap_adjusted.append(implemente_multiple_time(new_heatmap_row
                                                          , value= 0
                                                          , times= diff_len))

            # Update index of first heatmap value to take (associated to a token)
            step_index = max_token

        # Be sure, the last list of token get threshold num of value
        diff_len= threshold - len(token_adjusted[-1])
        token_adjusted[-1]= token_adjusted[-1] + [""]*diff_len

    return np.array(token_adjusted), np.array(heatmap_adjusted)


def plot_bar_heatmap(heatmap, figsize= (8, 1), cmap= 'Greens'):
    fig= plt.figure(figsize= figsize)

    color_map= plt.cm.get_cmap(cmap)
    # reversed_color_map = color_map.reversed()
    plt.imshow(heatmap[np.newaxis, :]
               , cmap=color_map, aspect="auto")
    plt.clim(0, 1)
    plt.show()
    pass


def plot_text_and_heatmap(text, heatmap, figsize= (5, 5), cmap= 'Greens'
                          , alpha= 0.7, word_or_letter= "letter"):
    # TODO: heatmap between -1 1, or 0 1 change vmin vmax imshow
    # if letter, then text is str
    # if word, text is a List[str] (tokens)
    if word_or_letter not in ["word", "letter"]:
        raise ValueError()

    color_map = plt.cm.get_cmap(cmap)

    # reshape to data to fit into a grid plot
    if word_or_letter == "letter":
        sentence_adjusted, heatmap_adjusted = reshape_text_to_plot(text=text
                                                              , heatmap=heatmap)
    else:
        sentence_adjusted, heatmap_adjusted = reshape_list_to_plot(text=text
                                                                   , heatmap=heatmap)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap grid
    im= ax.imshow(heatmap_adjusted, cmap=color_map, alpha= alpha, vmax= 1, vmin= -1)
    ax.set_yticks(range(heatmap_adjusted.shape[0]))  # data.shape[0]
    ax.set_xticks(range(heatmap_adjusted.shape[1]))  # data.shape[1]

    # Plot letters or words
    if word_or_letter == "letter":

        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        gridpoints = list(itertools.product(yticks, xticks))

        for i in gridpoints:
            plt.text(x=i[1], y=i[0], s=sentence_adjusted[i]
                     , fontsize='xx-small'
                     , weight="bold"
                     , verticalalignment= "center"
                     , horizontalalignment= "center" )


    else:
        x_localisation_per_row= []
        for row_token in sentence_adjusted:
            # We compute the x-postion for each token for each row
            cum_mean_token_per_row= cum_mean_token([len(x) for x in row_token])
            # Round like a boss
            x_localisation_per_row.append([(int(loc), cum_mean_token_per_row.index(loc))
                                           for loc in cum_mean_token_per_row])

        # Create the localisation combination
        gridpoints= []
        row= 0
        for x_loc_per_row in x_localisation_per_row:
            # row for y-axis, i for x-axis
            gridpoints += [[row, i, j] for i, j in x_loc_per_row]
            row += 1

        for i in gridpoints:
            plt.text(x=i[1], y=i[0], s=sentence_adjusted[i[0], i[2]]
                     , fontsize='xx-small'
                     , weight="bold"
                     , verticalalignment= "center"
                     , horizontalalignment= "center" )

    fig.colorbar(im, aspect= 75, orientation="horizontal")
    plt.axis('off')
    plt.show()
    pass


# # From input rebuild the text
# rebuild_sentence= rebuild_text(test_sentence[0].squeeze(0))
# # Resize heatmap
# heatmap_match_sentence_size_invert= prepare_heatmap(heatmap= heatmap
#                                                     , text= rebuild_sentence)
#
# # Plot
# plot_bar_heatmap(heatmap_match_sentence_size_invert)
# plot_text_and_heatmap(text= rebuild_sentence
#                       , heatmap= heatmap_match_sentence_size_invert
#                       , figsize=(7, 7))



# Test TODO :  generalize heatmap

output= cpu_model(test_sentence[0])

grd= cpu_model.activations_grad_cam(test_sentence[0], num_class= 0) # batch_size x num_features x feature_map_length
pooled_grd= torch.mean(grd, dim= [0, 2])

activations= cpu_model.get_activations(test_sentence[0]).detach()

for i in range(activations.shape[1]):
            activations[:, i, :] *= pooled_grd[i]

heatmap = torch.mean(activations, dim=1).squeeze().numpy()

heatmap_min= np.min(heatmap)
heatmap_normalized= (2.0*(heatmap - heatmap_min)/np.ptp(heatmap)) - 1
# heatmap_normalized = np.maximum(heatmap, 0)
# heatmap_normalized /= np.max(heatmap_normalized)


# From input rebuild the text
rebuild_sentence= rebuild_text(test_sentence[0].squeeze(0))
# Resize heatmap
heatmap_match_sentence_size_invert= prepare_heatmap(heatmap= heatmap_normalized
                                                    , text= rebuild_sentence)

# Plot
plot_bar_heatmap(heatmap_match_sentence_size_invert)

plot_text_and_heatmap(text= rebuild_sentence
                      , heatmap= heatmap_match_sentence_size_invert
                      , figsize=(7, 7)
                      , cmap= "RdYlGn")


## Test tokenize
def get_index_split_pattern(text):
    pattern_token_split = r'[\s\.\,\:\;\(\)\[\]\&\!\?]+'

    index_list= []
    for m in re.finditer(pattern_token_split, text):
        index_list.append((m.start(), m.end()))
    return index_list

# text_test= 'Hello boys! This is a fucking!! test.'
text_test= rebuild_sentence # not a shadow copy !!!!!!!
heatmap_test= np.copy(heatmap_normalized)
# heatmap_test= np.random.randint(0, 11, len(text_test))

index_sentence_token= get_index_split_pattern(text_test)

def implemente_multiple_time(base_list, value, times):
    return base_list + [value] * times


# @tailrec
def compute_heatmap_pooling(heatmap, type= "mean"):

    if type not in ["mean", "median", "max", "min", "maxabs", "minabs"]:
        raise ValueError()

    if type == "mean":
        return np.mean(heatmap)
    elif type == "median":
        return np.median(heatmap)
    elif type == "max":
        return np.max(heatmap)
    elif type == "min":
        return np.min(heatmap)
    elif type == "maxabs":
        return compute_heatmap_pooling(np.abs(heatmap), type= "max")
    elif type == "minabs":
        return compute_heatmap_pooling(np.abs(heatmap), type= "min")
    pass

step_index= 0
get_token= []
get_heatmap= []
get_extended_heatmap= []
sentence_len = len(text_test)
type_test= "max"

for i in index_sentence_token:
    if i[0] == 0:
        # If the first character is a token separator
        get_token.append(text_test[0:i[1]])
        value= compute_heatmap_pooling(heatmap_test[0:i[1]], type= type_test)
        get_heatmap.append(value)

        get_extended_heatmap= implemente_multiple_time(
            base_list= get_extended_heatmap
            , value= value
            , times= i[1] + 1
        )

        step_index = i[1]

    # # Append tokens
    # get_token.append(text_test[step_index:i[0]])
    # get_heatmap.append(np.mean(heatmap_test[step_index:i[0]]))
    # # Append token separators
    # get_token.append(text_test[i[0]:i[1]])
    # get_heatmap.append(np.mean(heatmap_test[i[0]:i[1]]))

    # Append tokens and token separators
    get_token.append(text_test[step_index:i[1]])
    value= compute_heatmap_pooling(heatmap_test[step_index:i[1]], type= type_test)
    get_heatmap.append(value)

    get_extended_heatmap = implemente_multiple_time(
        base_list=get_extended_heatmap
        , value=value
        , times= i[1] - step_index
    )

    # Update beginning of the next token
    step_index = i[1]

    # After the last token separator, if there is a token, add it
    if (index_sentence_token[-1] == i) & (text_test[step_index:] != '') :
        get_token.append(text_test[step_index:])
        value= compute_heatmap_pooling(heatmap_test[step_index:], type= type_test)
        get_heatmap.append(value)

        get_extended_heatmap = implemente_multiple_time(
            base_list=get_extended_heatmap
            , value=value
            , times= len(text_test[step_index:])
        )

# get_token= np.array(get_token)
get_heatmap= np.array(get_heatmap)
get_extended_heatmap= np.array(get_extended_heatmap)

plot_text_and_heatmap(text= rebuild_sentence
                      , heatmap= get_extended_heatmap
                      , figsize=(7, 7)
                      , cmap= "RdYlGn")



tested, _= reshape_list_to_plot(text= get_token, heatmap= get_heatmap)

def cum_mean_token(arr):

    stock_loc_token= 0
    cum_mean_position= []
    for i in range(len(arr)):
        if i == 0:
            # add the middle position for the first token
            cum_mean_position.append(arr[i]/2)
            stock_loc_token += arr[i]
        else:
            cum_mean_position.append(stock_loc_token + arr[i]/2)
            # Update by adding the arr[i] (token length)
            stock_loc_token += arr[i]

    return cum_mean_position



plot_text_and_heatmap(text= rebuild_sentence
                      , heatmap= heatmap_match_sentence_size_invert
                      , figsize=(7, 7)
                      , cmap= "RdYlGn")

plot_text_and_heatmap(text= get_token
                      , heatmap= get_heatmap
                      , figsize=(7, 12)
                      , cmap= "RdYlGn"
                      , word_or_letter= "word")
