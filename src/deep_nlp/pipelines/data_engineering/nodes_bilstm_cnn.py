import numpy as np
import nltk
from nltk.corpus import stopwords
import torch
from torch.utils.data import TensorDataset
import gensim

def tokenizer(phrase):
    phrase = str(phrase.lower())
    words = nltk.word_tokenize(phrase, language='french')
    stop_words = set(stopwords.words('french'))
    words_sans_ponct = [word for word in words if word.isalpha()]
    words_sans_ponct_sans_stopwords = [word for word in words_sans_ponct if not word in stop_words]
    return words_sans_ponct_sans_stopwords

def tokenizer_dataset(dataset):
    dataset['review'] = dataset["review"].apply(tokenizer)

    return dataset

def create_vocab(dataset):
    vocab = {}
    iter = 1

    for i in dataset['review']:
        for j in range(0, len(i)):
            if i[j] not in vocab.keys():
                vocab[i[j]] = iter
                iter += 1

    vocab["Unknown"] = iter

    return vocab

def word2index(s, vocab):
    a = []
    for j in range(0, len(s)):
        if s[j] in vocab.keys():
            a.append(vocab[s[j]])
        else:
            a.append(vocab["Unknown"])
    return a

def pad_input(sentence, sentence_size):
    return sentence[0:sentence_size] + (sentence_size-len(sentence))*[0]

def word2index_padding_dataset(dataset, vocab, sentence_size):
    dataset['review'] = dataset['review'].apply(word2index, args = (vocab, ))
    dataset['review'] = dataset['review'].apply(pad_input, args = (sentence_size, ))

    return dataset

def load_word2vec(word2vec_path):
    return gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True, unicode_errors='ignore')

def create_embed_matrix(w2v, vocab) :

    size_of_vocabulary = len(vocab) + 1

    embedding_matrix = np.zeros((size_of_vocabulary, w2v.vector_size))

    for word, i in vocab.items():

        try:
            embedding_vector = w2v.word_vec(word)
        except:
            embedding_vector = None
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

def to_array(sentences, sentence_size):
    features = np.zeros((len(sentences), sentence_size),dtype=int)
    for i in range(0, len(sentences)):
        features[i] = sentences[i]
    return features

def tensor_dataset(dataset, sentence_size):
    dataset_review = to_array(dataset["review"], sentence_size)
    dataset_label = np.array(dataset["label"], dtype=int)

    final_dataset = TensorDataset(torch.from_numpy(dataset_review), torch.from_numpy(dataset_label))
    return final_dataset