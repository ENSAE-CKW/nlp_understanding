import numpy as np

def order_tokens_by_importance(heatmap, tokens, threshold):
    best_word_explanation_index_one = np.where(heatmap >= threshold)[0]
    best_word_explanation_one = tokens[best_word_explanation_index_one]

    # Sort by the value (heatmap importance)
    sort_per_explanation_index_one = np.argsort(best_word_explanation_one)[::-1]  # to get the highest first
    best_word_explanation_one = best_word_explanation_one[sort_per_explanation_index_one]
    return best_word_explanation_one