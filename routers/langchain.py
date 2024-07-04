import os
import torch
from uuid import uuid4
from dotenv import load_dotenv
# from langchain_community.llms import HuggingFaceHub
# from langchain_community.llms import HuggingFacePipeline
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# .env HUGGINGFACEHUB_API_TOKEN 불러오기
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# 음성 인식 결과를 저장할 Chroma DB 생성
embeddings = HuggingFaceEmbeddings()
vectordb = Chroma(embedding_function=embeddings, persist_directory="./db")

# Model Import
print("[langchain] Torch CUDA Available : ", torch.cuda.is_available())

device = 0 if torch.cuda.is_available() else -1
if device==0: torch.cuda.empty_cache()

model_id = "llama3"

print("[langchain] Importing LLM Model :", model_id)
llm = ChatOllama(model=model_id)
print("[langchain]-[" + model_id + "]", llm.invoke("Hello World!"))
print("[langchain] Imported LLM Model :", model_id)

def process_speech_to_text(connection_uuid, converted_text):
    # 이전 음성 인식 결과 검색
    docs = vectordb.similarity_search(converted_text, k=3)
    context = " ".join([doc.page_content for doc in docs])

    # 텍스트 수정을 위한 프롬프트 템플릿
    correction_template = f"""
    이전 음성 인식 결과:
    {context}

    현재 음성 인식 결과:
    {converted_text}

    실시간 음성인식 결과를 더욱 매끄럽게 하기 위해 위 문장에 기반하여 아래 조건의 작업을 수행해주세요.
    1. 이전 결과를 고려하여 현재 텍스트를 문법적으로 올바르게 수정해주세요.
    2. 현재 음성 인식 결과에서 잡음이나 인식 오류를 제거해주세요.
    3. 이전 음성 인식 결과과의 문맥을 고려하여 자연스럽게 연결되도록 현재 음성 인식 결과를 수정해주세요.
    4. 답변은 한국어로 번역해주세요.
    4-1. 단, 음성인식 결과에 타 언어로 된 전문용어가 들어가 있다면 한국어로 변역하지 말아주세요.
    5. 추가적인 설명 없이 수정된 현재 음성 인식 결과만 제공해주세요.
    """

    # PromptTemplate :  원시 사용자 입력을 더 나은 입력으로 변환
    # OutputParser : 채팅 메시지를 문자열로 변환하는 출력 구문 분석기
    prompt1 = ChatPromptTemplate.from_template("[{korean_input}] translate the question into English. Don't say anything else, just translate it.")
    chain1 = (
        prompt1 
        | llm 
        | StrOutputParser()
    )

    prompt2 = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful, professional assistant in korean university. answer the question in Korean"),
        ("user", "{input}")
    ])
    chain2 = (
        {"input": chain1}
        | prompt2
        | llm
        | StrOutputParser()
    )

    corrected_text = chain2.invoke({"korean_input":correction_template})
    print(corrected_text)

    # 수정된 텍스트를 Chroma DB에 저장
    vectordb.add_texts(
        texts=[corrected_text],
        metadatas=[{"connection_uuid": connection_uuid}],
        ids=[str(uuid4())],
    )

    return corrected_text


def test():
    print("HF LLM Test")
    # 예시 사용
    connection_uuid = "example_connection_uuid"

    # 첫 번째 음성 인식 결과 처리
    # converted_text_1 = "이것은 시장 경제에 대한 설명입니다."
    # process_speech_to_text(connection_uuid, converted_text_1)

    # 두 번째 음성 인식 결과 처리
    converted_text_2 = "시장 경제는 가격 아아아아아아아아아 기구를 통해 자원을 배분하는 경제 체제입니다."

    return process_speech_to_text(connection_uuid, converted_text_2)
