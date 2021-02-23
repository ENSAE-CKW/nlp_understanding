from kedro.pipeline import Pipeline, node
from .nodes_bilstm_cnn import *


def create_bilstm_cnn_pipeline_de(**kwargs):
    return Pipeline(
        [
            node(
                func=tokenizer_dataset
                , inputs = ["allocine_train"]
                , outputs = "train_tokenized"
                , name = "train tokenization"
                , tags = ["bilstm_cnn","train_set", "tokenizer"]
                ),
            node(
                func=tokenizer_dataset
                , inputs=["allocine_valid"]
                , outputs="valid_tokenized"
                , name="valid tokenization"
                , tags=["bilstm_cnn", "train_set", "tokenizer"]
            ),
            node(
                func=tokenizer_dataset
                , inputs=["allocine_test"]
                , outputs="test_tokenized"
                , name="test tokenization"
                , tags=["bilstm_cnn", "train_set", "tokenizer"]
            ),
            node(
                func=create_vocab
                , inputs=["train_tokenized"]
                , outputs="vocab"
                , name="vocabulary creation"
                , tags=["bilstm_cnn", "vocabulary"]
            ),
            node(
                func=word2index_padding_dataset
                , inputs=["train_tokenized", "vocab", "params:sentence_size"]
                , outputs="index_pad_train_dataset"
                , name="word to index and padding train dataset"
                , tags=["bilstm_cnn", "word to index","padding", "train"]
            ),
            node(
                func=word2index_padding_dataset
                , inputs=["valid_tokenized", "vocab", "params:sentence_size"]
                , outputs="index_pad_valid_dataset"
                , name="word to index and padding valid dataset"
                , tags=["bilstm_cnn", "word to index", "padding", "valid"]
            ),
            node(
                func=word2index_padding_dataset
                , inputs=["test_tokenized", "vocab", "params:sentence_size"]
                , outputs="index_pad_test_dataset"
                , name="word to index and padding test dataset"
                , tags=["bilstm_cnn", "word to index", "padding", "test"]
            ),
            node(
                func=load_word2vec
                , inputs=["params:bilstm_word2vec_path"]
                , outputs="word2vec_bilstm"
                , name="load word2vec"
                , tags=["bilstm_cnn", "word2vec"]
            ),
            node(
                func=create_embed_matrix
                , inputs=["word2vec_bilstm", "vocab"]
                , outputs="embed_matrix"
                , name="embedding matrix creation"
                , tags=["bilstm_cnn", "embedding matrix"]
            ),
            node(
                func=tensor_dataset
                , inputs=["index_pad_train_dataset", "params:sentence_size"]
                , outputs="final_train_dataset"
                , name="dataset train final"
                , tags=["bilstm_cnn", "train", "final dataset"]
            ),
            node(
                func=tensor_dataset
                , inputs=["index_pad_valid_dataset", "params:sentence_size"]
                , outputs="final_valid_dataset"
                , name="dataset valid final"
                , tags=["bilstm_cnn", "valid", "final dataset"]
            ),
            node(
                func=tensor_dataset
                , inputs=["index_pad_test_dataset", "params:sentence_size"]
                , outputs="final_test_dataset"
                , name="dataset test final"
                , tags=["bilstm_cnn", "test", "final dataset"]
            )
        ])