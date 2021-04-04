from kedro.pipeline import Pipeline, node

from .nodes_bilstm_cnn import run_train
from .nodes_bilstm_cnn import prepare_batch
from .nodes_bilstm_cnn import save_model

def create_bilstm_cnn_pipeline_ds(**kwargs):
    return Pipeline(
        [   node(
                func = prepare_batch,
                inputs = ["final_train_dataset", "final_valid_dataset", "final_test_dataset", "params:bilstm_batch_size"],
                outputs = ["train_batch", "valid_batch", "test_batch"],
                tags=["bilstmcnn", "training", "batch"],
                name = "batch preparation"
            ),
            node(
                func = run_train,
                inputs = ["params:cnn_cuda_allow", "train_batch", "valid_batch", "params:bilstm_num_epochs"
                    , "params:bilstm_patience", "params:bilstm_lr", "embed_matrix", "params:sentence_size"
                    , "params:bilstm_input_dim", "params:bilstm_hidden_dim", "params:bilstm_layer_dim"
                    , "params:bilstm_output_dim", "params:bilstm_feature_size", "params:bilstm_kernel_size"
                    , "params:bilstm_dropout_rate"],
                outputs = "bilstmcnn_model",
                tags= ["train", "bilstmcnn"],
            ),
            node(
                func=save_model,
                inputs="bilstmcnn_model",
                outputs="bilstm_cnn_model_for_save"
            ),
        ]
    )