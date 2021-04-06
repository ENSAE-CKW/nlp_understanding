import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow import log_artifact
sns.set()

# Confusion matrix
def plot_cm(target_all, predictions_all
            , path= "data/08_reporting/confusion_matrix.png", show= False):
    data_cm = metrics.confusion_matrix(target_all, predictions_all)

    positif_negatif_dict_map = {1: "positif", 0: "negatif"}

    df_cm = pd.DataFrame(data_cm, columns=[positif_negatif_dict_map[i] for i in np.unique(target_all)]
                         , index=[positif_negatif_dict_map[i] for i in np.unique(target_all)])

    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'

    plt.figure(figsize=(7, 6))

    sns.heatmap(df_cm, cmap="Blues", annot=True, fmt='g')

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    log_artifact(path)

    if show:
        plt.show()

    pass