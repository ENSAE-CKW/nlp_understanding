from ..utils.utils import resize_array

def prepare_heatmap(heatmap, text):
    sentence_len = len(text)
    # Cut the heatmap to the sentence len (character level)
    resize_heatmap= resize_array(heatmap, sentence_len)

    # The first character is the last in the input model
    # So we need to invert our heatmap
    return resize_heatmap[::-1]