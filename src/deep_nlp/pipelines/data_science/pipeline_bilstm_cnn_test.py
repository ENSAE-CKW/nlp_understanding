from kedro.pipeline import Pipeline, node

from .nodes_bilstm_cnn import bilstm_test


def create_bilstm_cnn_test(**kwargs):
    return Pipeline(
        [
            node(
                func= bilstm_test
                , inputs= ["params:cnn_cuda_allow", "embed_matrix", "params:sentence_size"
                    , "params:bilstm_input_dim", "params:bilstm_hidden_dim", "params:bilstm_layer_dim"
                    , "params:bilstm_output_dim", "params:bilstm_feature_size", "params:bilstm_kernel_size"
                    , "params:bilstm_dropout_rate", "params:bilstm_padded", "test_batch", "bilstm_cnn_model_for_save"
                    , "vocab_bilstm", "params:bilstm_index_nothing"]
                , outputs= None
                , tags= ["bilstm_cnn_test", "test"]
            )
        ]
    )
