import numpy as np
from ..utils import implemente_multiple_time


def reshape_token_to_plot(text, heatmap, threshold= 10):
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