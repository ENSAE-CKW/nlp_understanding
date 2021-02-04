from kedro.pipeline import Pipeline, node
from .nodes_logistic_bow import train, evaluate


def create_logistic_bow_pipeline_ds(**kwargs):
    return Pipeline(
        [
            node(
                func= train
                , inputs= ["train_data_logistic_bow", "train_y_logistic_bow"]
                , outputs= ["logistic_bow_model_train", "logistic_bow_scaler"] # model
                , tags= ["logistic_bow_train", "train"]
            ),
            node(
                func=evaluate
                , inputs=["valid_data_logistic_bow", "valid_y_logistic_bow", "logistic_bow_model_train"
                          , "logistic_bow_scaler"]
                , outputs= None
                , tags=["logistic_bow_train", "train"]
            )
        ]
    )
