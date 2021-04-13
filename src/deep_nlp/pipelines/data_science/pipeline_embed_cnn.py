
from kedro.pipeline import Pipeline, node

from src.deep_nlp.pipelines.data_science.nodes_embed_cnn import *


def create_embed_cnn_pipeline_ds(**kwargs):
    return Pipeline(
        [
            node(
                func = init_model,
                inputs = ["embed_for_torch","params:sentence_size","params:embcnn_nb_filtre", "params:embcnn_type_filtre","params:embcnn_nb_output","params:embcnn_dropout","params:embcnn_padded"],
                outputs = "model",
                tags = ["embed_cnn","training", "model"],
                name = "initiailisation model"
            ),
            node(
                func = creation_batch,
                inputs = ["allocine_train_inter", "allocine_valid_inter","allocine_test_inter","params:device","params:batch_size"],
                outputs = ["train_batch_embed", "valid_batch_embed", "test_batch_embed"],
                tags=["embed_cnn", "training", "batch"],
                name = "batch initialisation"
            ),
            node(
                func = run_model,
                inputs = ["model","params:embcnn_n_epochs", "params:device", "train_batch_embed","valid_batch_embed"],
                outputs = "embed_cnn_model",
                tags= ["embed_cnn", "training", "model"],
                name= "model's run"
            ),
            node(
                func=save_model,
                inputs="embed_cnn_model",
                outputs="embed_cnn_model_for_save"
            )

        ]
    )
