import numpy as np

def reshape_character_to_plot(text, heatmap, threshold= 40):
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
            text_adjusted= np.array(list(text)).reshape((threshold, entire_part))
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