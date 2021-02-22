import transformers
from transformers import CamembertTokenizer
from deep_nlp.bert_cnn import bertToTensor
import pandas as pd


def load_camembert(bertcnn_bert_path: str) -> transformers.PreTrainedTokenizerFast:
    camembert= CamembertTokenizer.from_pretrained(bertcnn_bert_path)
    return camembert

def load_dataloader(data: pd.DataFrame, max_seq_len: int
                    , tokenizer: transformers.PreTrainedTokenizerFast) -> bertToTensor:
    return bertToTensor(data_df= data, max_seq_len= max_seq_len, tokenizer= tokenizer)