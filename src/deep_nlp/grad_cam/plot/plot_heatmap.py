import matplotlib.pyplot as plt
import numpy as np
import itertools

from ..utils.letter import reshape_character_to_plot
from ..utils.token import reshape_token_to_plot


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
                          , alpha= 0.7, word_or_letter= "letter", threshold= None, fontsize_text= "xx-small"
                          , force_color= True):
    # if letter, then text is str
    # if word, text is a List[str] (tokens)
    if word_or_letter not in ["word", "letter"]:
        raise ValueError()

    color_map = plt.cm.get_cmap(cmap)

    # reshape to data to fit into a grid plot
    if word_or_letter == "letter":
        if threshold is None:
            threshold= 40
        sentence_adjusted, heatmap_adjusted = reshape_character_to_plot(text=text
                                                                        , heatmap=heatmap
                                                                        , threshold= threshold)
    else:
        if threshold is None:
            threshold= 10
        sentence_adjusted, heatmap_adjusted = reshape_token_to_plot(text=text
                                                                   , heatmap=heatmap
                                                                   , threshold= threshold)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap grid
    vmax= 1
    vmin= -1 if np.min(heatmap_adjusted) < 0 else 0 # if there is neg value, put a -1
    # else 0

    # if (color_map != "Greens") :
    #     vmin= -1

    # if (color_map == "Greens") :
    #     vmin= 0


    im= ax.imshow(heatmap_adjusted, cmap=color_map, alpha= alpha, vmax= vmax, vmin= vmin)

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
                     , fontsize= fontsize_text#'xx-small'
                     , weight="bold"
                     , verticalalignment= "center"
                     , horizontalalignment= "center")

    cbar= fig.colorbar(im, aspect= 100, orientation="horizontal")
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    plt.axis('off')
    plt.show()
    pass


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