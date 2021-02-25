from kedro.pipeline import Pipeline, node

from .nodes_bilstm_cnn import bilstm_test

def create_bilstm_cnn_test(**kwargs):
    return Pipeline(
        [
            node(
                func= bilstm_test
                , inputs= ["bilstmcnn_model", "params:cnn_cuda_allow", "test_batch"]
                , outputs= None
                , tags= ["bilstm_cnn_test", "test"]
            )
        ]
    )