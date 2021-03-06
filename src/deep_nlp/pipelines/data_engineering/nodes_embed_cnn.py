from typing import Any, Dict, List, Tuple, Set
import numpy as np
import spacy
import pandas as pd
import torch
import subprocess
import gensim

#TODO : add types to args in function
def creation_nlp():
    subprocess.run("python -m spacy download fr_core_news_sm")
    return spacy.load("fr_core_news_sm",  disable=["tagger", "parser","ner"])

def token_sentence(sentence: str, nlp: spacy.language.Language) -> List[str]:
    return [X.lemma_ for X in nlp(sentence) if X.is_alpha & (not (X.is_stop))]

def token_df(df: pd.DataFrame, col_name: str,nlp : spacy.language.Language) -> pd.DataFrame:
    df["tokenization"] = df.apply(lambda x: token_sentence(x[col_name],nlp), axis = 1) #.lower()
    return df

#we assume that df_tokenised has a column called "tokenization"
def vocab(df_tokenised: pd.DataFrame) -> Set[str]:
    tokenization = df_tokenised["tokenization"]
    all_words = [element for list in tokenization for element in list]
    return set(all_words)

#We assume that the embedding is an csv file with
#name : name of the word
#columns named from 0 à D-1 where D stands for the dimension of the embedding
def vectors_embed(path_embed):
    model = gensim.models.KeyedVectors.load_word2vec_format(path_embed, binary=True,unicode_errors='ignore')
    vectors = pd.DataFrame(model.wv.syn0)
    name = [word for word in model.wv.vocab]
    vectors["name"] = name
    return name,vectors

#return dic with word in embed and vocab
#with a index
def words_index(name_embed, vocab):
    word_with_vectors = vocab.intersection(name_embed)
    return dict(zip(word_with_vectors, range(len(word_with_vectors))))

#vectors describes vectors of embed
def index_vectors_embed(vectors, word_index_dict):
    vectors["index"] = vectors.apply(lambda x: word_index_dict.get(x["name"]), axis=1)
    vectors.sort_values(by = "index")
    return vectors

def token_to_index(sentence,word_index_dict):
    return [word_index_dict.get(mots, None) for mots in sentence]

def token_to_index_df(df_tokenised, word_index_dict):
    tokenization = list(df_tokenised["tokenization"])
    indexed = [token_to_index(sentence, word_index_dict) for sentence in tokenization]
    df_tokenised["indexed"] = indexed
    return df_tokenised

def pad(list_word, nb):
    if len(list_word) >= nb:
        res = list_word[:nb]
    else:
        to_add = list(np.repeat(-1,nb - len(list_word)))
        res = list_word + to_add
    return res

def pad_df(df_indexed, nb):
    df_indexed["padded"] = df_indexed.apply(lambda x: pad(x["indexed"], nb), axis = 1)
    return df_indexed

def reshape_df(df_padded, embed_torch):
    df = pd.DataFrame(list(df_padded["padded"]))
    nb_row = embed_torch.shape[0] - 2
    df = df.replace([np.nan,-1], [nb_row, nb_row + 1])
    df["label"] = df_padded["label"]
    return df

def embed_to_torch(embed):
    vec = embed.drop(columns=["name","index"])
    tens = torch.from_numpy(vec.to_numpy())
    pad_unk = torch.rand([2,tens.shape[1]])
    tens = torch.cat([tens,pad_unk])
    return tens

def save_embed(embed):
    return pd.DataFrame(embed.numpy())
