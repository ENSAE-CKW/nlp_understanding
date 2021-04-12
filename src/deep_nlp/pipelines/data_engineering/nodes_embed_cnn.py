from typing import Any, Dict, List, Tuple, Set
import numpy as np
import spacy
import pandas as pd
import torch
import subprocess
import gensim

from nltk.corpus import stopwords
import unicodedata
import re

#TODO : add types to args in function
def creation_nlp(all_stop_word = False):
    subprocess.run("python -m spacy download fr_core_news_sm")
    nlp = spacy.load("fr_core_news_sm",  disable=["tagger", "parser","ner"])
    if not all_stop_word:
        nlp.Defaults.stop_words -= {"apres", "assez", "importe", "moindres", "moins", "n'", "n’", "ne", "neanmoins",
                                    "ni",
                                    "nombreuses", "nul", "sans", "sauf", "stop", "suffisant", "tellement", "pas"}
    return nlp


def clean_tokens(token: str, pattern= r'[\s\.\,\:\;\"\'\(\)\[\]\&\!\?\/\\]+'):
    cleaned_token= re.sub(pattern, "", token)
    return ''.join(c for c in unicodedata.normalize('NFD', cleaned_token)
                        if unicodedata.category(c) != 'Mn').lower()

def strip_accents_and_lowercase(token: str, all_stop_word= False) -> str:
    if all_stop_word:
        french_stopwords = tuple([clean_tokens(word) for word in stopwords.words('french')])
    else:
        french_stopwords = tuple([])
    lower_cleaned_token= clean_tokens(token)
    if lower_cleaned_token not in french_stopwords:
        return lower_cleaned_token
    else:
        return ""

def last_clean(text: str) -> str:
    # From https://stackoverflow.com/a/29920015/5909675
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', text.replace("#", " "))
    return " ".join([m.group(0) for m in matches])

def token_sentence(sentence: str, nlp: spacy.language.Language) -> List[str]:
    return [str(X.lemma_) for X in nlp(sentence) if X.is_alpha & (not (X.is_stop))]

def token_df(df: pd.DataFrame, col_name: str,nlp : spacy.language.Language) -> pd.DataFrame:
    # Clean token mano
    pattern = r'[\s\.\,\:\;\(\)\[\]\&\!\?\/\\]+'

    df["intermediate"] = df[col_name].apply(lambda sentence: re.split(pattern, sentence))

    df["intermediate"] = df["intermediate"].apply(
        lambda list_of_token: [clean_tokens(token)
                               for token in list_of_token])

    df["intermediate"] = df["intermediate"].apply(
        lambda list_of_token: [strip_accents_and_lowercase(token)
                               for token in list_of_token])

    df["intermediate"] = df["intermediate"].apply(lambda list_of_token: [last_clean(token)
                                                                         for token in list_of_token if
                                                                         token != ""])

    df[col_name] = df["intermediate"].apply(
        lambda list_of_token: re.sub(r'\s{2,}', "", " ".join(list_of_token))
    )

    pop_intermediate= df.pop("intermediate")
    del pop_intermediate

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
