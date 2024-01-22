import torch
from torch import nn
from transformers import BertModel, BertTokenizer
import json
import nltk
import pandas as pd
from konlpy.tag import Okt
import networkx as nx

nltk.download("punkt")


# BertSumLSTM 모델 정의
class BertSumLSTM(nn.Module):
    def __init__(self, bert_model_name, hidden_dim, num_layers, dropout):
        super(BertSumLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.lstm = nn.LSTM(
            self.bert.config.hidden_size,
            hidden_dim,
            num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        lstm_output, _ = self.lstm(sequence_output)
        logits = self.classifier(lstm_output).squeeze(-1)
        return logits


# 중요한 문장 추출 함수
def extract_important_sentences(text, model, tokenizer, num_sentences=2):
    model.eval()
    with torch.no_grad():
        input_ids, attention_masks, sentences = create_input(text, tokenizer)
        logits = model(input_ids, attention_masks)
        scores = torch.sigmoid(logits).squeeze().tolist()

        important_sentence_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:num_sentences]
        important_sentences = [sentences[i] for i in important_sentence_indices]
        return important_sentences


# 입력 생성 함수
def create_input(text, tokenizer):
    sentences = nltk.sent_tokenize(text)
    input_ids = []
    attention_masks = []

    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=35,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids.append(encoded_dict["input_ids"])
        attention_masks.append(encoded_dict["attention_mask"])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks, sentences


# 명사 추출 함수
def extract_nouns(text, stopwords):
    okt = Okt()
    nouns = okt.nouns(text)
    filtered_nouns = [noun for noun in nouns if noun not in stopwords and len(noun) > 1]
    return filtered_nouns


# 텍스트 랭크 알고리즘 적용 함수
def apply_text_rank(nouns):
    graph = nx.Graph()
    graph.add_nodes_from(nouns)

    for i in range(len(nouns)):
        for j in range(i + 1, len(nouns)):
            graph.add_edge(nouns[i], nouns[j])

    scores = nx.pagerank(graph)
    return scores


# 모델 및 토크나이저 초기화
bert_model_name = "bert-base-multilingual-cased"
hidden_dim = 256
num_layers = 2
dropout = 0.3
sentence_model = BertSumLSTM(bert_model_name, hidden_dim, num_layers, dropout)
tokenizer = BertTokenizer.from_pretrained(bert_model_name)

# 불용어 목록
stopwords = [
    "이",
    "그",
    "저",
    "것",
    "수",
    "등",
    "해",
    "있",
    "되",
    "없",
    "않",
    "같",
    "에서",
    "로",
    "고",
    "으로",
    "다",
    "만",
    "도",
    "의",
    "가",
    "이런",
    "저런",
    "합니다",
    "하세요",
]


# JSON 데이터 처리 함수
def process_json_data(json_data):
    df = pd.DataFrame(json_data)

    # 중요한 문장과 단어 추출
    important_sentences = []
    important_words_list = []

    for text in df["sumText"]:
        imp_sentences = extract_important_sentences(
            text, sentence_model, tokenizer, num_sentences=2
        )
        important_sentences.append(imp_sentences)

        nouns = extract_nouns(text, stopwords)
        scores = apply_text_rank(nouns)
        imp_words = sorted(scores, key=scores.get, reverse=True)[:5]
        important_words_list.append(imp_words)

    # 결과 저장
    df["important_sentence"] = important_sentences
    df["important_words"] = important_words_list

    return df


import pandas as pd
import numpy as np
from ast import literal_eval


class Example:
    def __init__(self, df_original, important_sentence, important_words):
        self.df_original = df_original
        self.important_sentence = self.tokenize_and_eval(important_sentence)
        self.important_words = self.tokenize_and_eval(important_words)

    def tokenize_and_eval(self, text):
        if isinstance(text, str):
            # 문자열을 공백을 기준으로 분할하고 리스트로 반환
            return text.split()
        else:
            # 이미 리스트 형태인 경우 그대로 반환
            return text

    def update_unProcessedText(self):
        l = []
        for text_list in self.df_original["unProcessedText"]:
            if text_list[0] in self.important_sentence[0]:
                l.append("highlight")
            elif text_list[0] in set(self.important_words):
                l.append("comment")
            else:
                l.append("none")
        for ind, unprocessedtext in enumerate(self.df_original["unProcessedText"]):
            unprocessedtext[1] = l[ind]

    def clear_other_rows(self):
        for col in self.df_original.columns:
            if col != "unProcessedText":
                self.df_original.loc[1:, col] = np.nan


# 추가적인 데이터 처리 함수
def additional_data_processing(processed_data):
    # Example 클래스를 사용하여 추가 처리
    example_instance = Example(
        processed_data,
        processed_data["important_sentence"],
        processed_data["important_words"],
    )
    example_instance.update_unProcessedText()
    example_instance.clear_other_rows()
    return example_instance.df_original
