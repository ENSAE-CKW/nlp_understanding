import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import List
from mlflow import log_artifact
sns.set()

def plot_barplot(list_of_token: List[str], num_to_display: int= 20, title: str= None
                 , path= "data/08_reporting/embed_cnn/barplot.png", show= False) -> None:
    w= dict(Counter(list_of_token).most_common(num_to_display))

    fig, ax = plt.subplots(figsize= (6, 5))
    ax.bar(w.keys(), w.values(), figure= None)

    ax.set_title(title)
    ax.xaxis.set_tick_params(rotation= 80)

    plt.tight_layout()
    plt.savefig(path, bbox_inches = "tight")
    log_artifact(path)

    if show:
        plt.show()

    pass

