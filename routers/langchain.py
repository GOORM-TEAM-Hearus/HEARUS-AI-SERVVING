import os
import re
import torch
import json
from uuid import uuid4
from dotenv import load_dotenv
# from langchain_community.llms import HuggingFaceHub
# from langchain_community.llms import HuggingFacePipeline
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma

# .env HUGGINGFACEHUB_API_TOKEN 불러오기
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# 음성 인식 결과를 저장할 Chroma DB 생성
embeddings = HuggingFaceEmbeddings()
vectordb = Chroma(embedding_function=embeddings, persist_directory="./db")

# Model Import
print("[LangChain] Torch CUDA Available : ", torch.cuda.is_available())

device = 0 if torch.cuda.is_available() else -1
if device==0: torch.cuda.empty_cache()

model_id = "llama3"

print("[LangChain] Importing LLM Model :", model_id)
llm = ChatOllama(model=model_id)
print("[LangChain]-[" + model_id + "]", llm.invoke("Hello World!"))
print("[LangChain] Imported LLM Model :", model_id)

def speech_to_text_modification(connection_uuid, converted_text):
    # 이전 음성 인식 결과 검색
    # 마지막 3개의 음성만을 가져온다
    docs = vectordb.max_marginal_relevance_search(converted_text, k=3)
    context = " ".join([doc.page_content for doc in reversed(docs)])
    print("[LangChain] Original Converted Text : ", converted_text)

    textData = f"""
    이전 음성 인식 결과:
    {context}

    현재 음성 인식 결과:
    {converted_text}
    """

    # PromptTemplate : 원시 사용자 입력을 더 나은 입력으로 변환
    # OutputParser : 채팅 메시지를 문자열로 변환하는 출력 구문 분석기
    prompt = ChatPromptTemplate.from_template("""
        {textData}
                                              
        실시간 음성인식 결과를 더욱 매끄럽게 하기 위해 위 문장에 기반하여 아래 조건의 작업을 수행해주세요.
        1. 이전 결과를 고려하여 현재 텍스트를 문법적으로 올바르게 수정해주세요.
        2. 현재 음성 인식 결과에서 잡음이나 인식 오류를 제거해주세요.
        3. 이전 음성 인식 결과의 문맥을 고려하여 자연스럽게 연결되도록 현재 음성 인식 결과를 수정해주세요.
        4. 답변은 한국어로 번역해주세요.
        4-1. 단, 음성인식 결과에 타 언어로 된 전문용어가 들어가 있다면 한국어로 변역하지 말아주세요.
        5. 추가적인 설명 없이 수정된 현재 음성 인식 결과만 출력해주세요.
        6. 문장에 부가 설명을 붙이지 말고 오로지 제공된 현재 음성인식 결과에서만 수정해주세요.

        개선된 문장만을 "result" key의 value에 담아 JSON 형태로 제공해주세요
        결과 외에는 그 어떤 텍스트도 답변하지 말아주세요
    """)

    chain1 = (
        prompt 
        | llm 
        | StrOutputParser()
    )

    # prompt2 = ChatPromptTemplate.from_messages([
    #     ("system", "You are a helpful, professional assistant in korean university. answer the question in Korean"),
    #     ("user", "{input}")
    # ])
    # chain2 = (
    #     {"input": chain1}
    #     | prompt2
    #     | llm
    #     | StrOutputParser()
    # )

    corrected_text = chain1.invoke({"textData" : textData})
    json_result = parse_JSON(corrected_text)

    result_value = json_result.get('result')
    if result_value:
        print("[LangChain]-[" + model_id + "] Result value:", result_value)
    else:
        print("[LangChain]-[" + model_id + "] No 'result' key found in the JSON")
        return None

    # Chroma DB에 데이터 저장
    vectordb.add_documents(
        documents=[Document(page_content=result_value, metadata={"connection_uuid": connection_uuid})],
        ids=[str(uuid4())],
    )

    return result_value

def delete_data_by_uuid(connection_uuid):
    # connection_uuid에 해당하는 데이터 삭제
    documents = vectordb.get(metadata={"connection_uuid": connection_uuid})
    
    # 문서에서 ID 추출
    document_ids = [doc.id for doc in documents]
    
    # 문서 ID에 해당하는 데이터 삭제
    if document_ids:
        vectordb.delete(ids=document_ids)
        vectordb.persist()
        print(f"[LangChain] Data with connection_uuid '{connection_uuid}' has been deleted from ChromaDB.")
    else:
        print(f"[LangChain] No data found with connection_uuid '{connection_uuid}' in ChromaDB.")


def parse_JSON(llm_response):
    json_pattern = re.compile(r'{[^{}]*?}')

    # LLM 응답에서 JSON 값 찾기
    json_match = json_pattern.findall(llm_response)
    
    if json_match:
        json_str = json_match[-1]
        
        try:
            json_data = json.loads(json_str)
            return json_data
        except json.JSONDecodeError:
            print("[LangChain]-[parse_JSON] Invalid JSON format")
            return None
    else:
        print("[LangChain]-[parse_JSON] No JSON found in the LLM response")
        return None