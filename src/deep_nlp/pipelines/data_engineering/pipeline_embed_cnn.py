from kedro.pipeline import Pipeline, node
from .nodes_embed_cnn import *


def create_embed_cnn_pipeline_de(**kwargs):
    return Pipeline(
        [
            node(
                func=token_df #verif inter
                , inputs = ["allocine_train","params:colname_allocine", "nlp"]
                , outputs = "train_tokenized"
                , name = "tokenization"
                , tags = ["embed_cnn","data_proc","train_set"]
                ),
            node(
                func = creation_nlp,
                inputs = None,
                outputs = "nlp",
                name = "Spacy Language initialisation",
                tags = ["embed_cnn","data_proc"]
            ),
            node(
                func=token_df
                , inputs=["allocine_test", "params:colname_allocine", "nlp"]
                , outputs="test_tokenized"
                , name="tokenization of test set"
                , tags=["embed_cnn", "data_proc", "test_set"]
            ),
            node(
                func=token_df
                , inputs=["allocine_valid", "params:colname_allocine", "nlp"]
                , outputs="valid_tokenized"
                , name="tokenization of valid set"
                , tags=["embed_cnn", "data_proc", "valid_set"]
            ),
            node(
                func=vocab
                , inputs = "train_tokenized"
                , outputs= "vocab_set"
                , name = "vocab initialisation"
                , tags = ["embed_cnn", "data_proc", "train"]
            ) ,
            node(
                func = vectors_embed
                , inputs = ["params:word2vec_path"]
                , outputs= ["name_word_in_embed", "vectors"]
                , tags=["embed_cnn", "data_proc", "train"]
                , name = "embedding initialisation"
            ),
            node(
                func = words_index
                , inputs = ["name_word_in_embed","vocab_set"]
                , outputs = "word_ind_dict"
                , tags=["embed_cnn", "data_proc", "train"]
                , name = "index for word from voc"
            ),
            node(
                func = token_to_index_df
                , inputs = ["train_tokenized", "word_ind_dict"]
                , outputs = "train_indexed"
                , tags=["embed_cnn", "data_proc", "train_set"]
                , name = "index for word from df train"
            ),
            node(
                func=token_to_index_df
                , inputs=["test_tokenized", "word_ind_dict"]
                , outputs="test_indexed"
                , tags=["embed_cnn", "data_proc", "test_set"]
                , name="index for word from df test"
            ),
            node(
                func=token_to_index_df
                , inputs=["valid_tokenized", "word_ind_dict"]
                , outputs="valid_indexed"
                , tags=["embed_cnn", "data_proc", "test_set"]
                , name="index for word from df valid"
            ),
            node(
                func = pad_df
                , inputs = ["train_indexed","params:sentence_size"]
                , outputs = "train_padded"
                , tags=["embed_cnn", "data_proc", "train_set"]
                , name = "padding df train"),
            node(
                func=pad_df
                , inputs=["test_indexed", "params:sentence_size"]
                , outputs="test_padded"
                , tags=["embed_cnn", "data_proc", "test_set"]
                , name="padding df test"),
            node(
                func=pad_df
                , inputs=["valid_indexed", "params:sentence_size"]
                , outputs="valid_padded"
                , tags=["embed_cnn", "data_proc", "valid_set"]
                , name="padding df valid"),
            node(
                func = reshape_df
                , inputs = ["train_padded","embed_for_torch"]
                , outputs = "allocine_train_short_inter"
                , tags = ["embed_cnn","data_proc", "train_set"]
                , name = "reformat train"),
            node(
                func=reshape_df
                , inputs=["test_padded","embed_for_torch"]
                , outputs="allocine_test_short_inter"
                , tags=["embed_cnn","data_proc", "test_set"]
                , name="reformat tests"),
            node(
                func=reshape_df
                , inputs=["valid_padded","embed_for_torch"]
                , outputs="allocine_valid_short_inter"
                , tags=["embed_cnn", "data_proc", "valid_set"]
                , name="reformat valid"),
            node(
                func = index_vectors_embed
                ,inputs = ["vectors","word_ind_dict"]
                ,outputs = "word2vec_for_save"
                ,tags = ["embed_cnn","data_proc","embedding"]
                , name = "save_embed"
            ),
            node(
                func = embed_to_torch
            , inputs = "word2vec_for_save"
            , outputs = "embed_for_torch"
            , tags = ["embed_cnn","data_proc","embedding"]
            , name = "embed as tensors"),
            node(
                func = save_embed,
            inputs = "embed_for_torch",
            outputs = "save_embedding",
            tags = ["embed_cnn","data_proc","embedding"])
        ]
    )