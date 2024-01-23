import torch
from torch import nn
from transformers import BertModel, BertTokenizer
import nltk
import pandas as pd
from konlpy.tag import Okt
import networkx as nx
import json

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


# JSON 파일 처리
def process_json_data(file_path):
    # JSON 파일을 데이터프레임으로 불러오기
    df = pd.DataFrame(file_path)

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

    df["important_sentence"] = important_sentences
    df["important_words"] = important_words_list

    return df


import pandas as pd
import json


def process_data_to_json(df, important_words, important_sentences):
    # df_unpro가 words에 속한다면 'comment', sentence에 속한다면 'highlight'로 변경
    df_unpro = df["unProcessedText"].tolist()
    for l in df_unpro:
        if l[0] in important_words:
            l[1] = "comment"
        elif l[0] in important_sentences:
            l[1] = "highlight"
        elif l[0] != "br":
            l[1] = "none"

    dictData = {}
    dictData["arrStart"] = df["arrStart"][0]
    dictData["arrEnd"] = df["arrEnd"][0]
    dictData["unProcessedText"] = df_unpro
    dictData["sumText"] = df["sumText"][0]

    return dictData


def save_json(data, file_path):
    # 데이터를 JSON 파일로 저장
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent="	")
