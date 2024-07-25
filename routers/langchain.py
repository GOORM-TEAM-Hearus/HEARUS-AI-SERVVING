import os
import re
import torch
import json
import asyncio
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.empty_cache()

print("[LangChain] Torch CUDA Available : ", torch.cuda.is_available())
print("[LangChain] Current Device : ", device)

model_id = "llama3"

print("[LangChain] Importing LLM Model :", model_id)
llm = ChatOllama(model=model_id, device=device)
print("[LangChain]-[" + model_id + "]", llm.invoke("Hello World!"))
print("[LangChain] Imported LLM Model :", model_id)

def parse_JSON(llm_response, is_array=False):
    json_pattern = re.compile(r'{[^{}]*?}')

    # LLM 응답에서 JSON 값 찾기
    json_match = json_pattern.findall(llm_response)
    
    if json_match and is_array:
        json_array = []
        for string in json_match:
            try:
                json_array.append(json.loads(string))
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {str(e)}")
                print(string)
        return json_array
    elif json_match:
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

######## STT LangChain ########
async def speech_to_text_modification(connection_uuid, converted_text):
    # 이전 음성 인식 결과 검색
    # 마지막 1개의 음성만을 가져온다
    docs = await asyncio.to_thread(
        vectordb.get,
        where={"connection_uuid": connection_uuid},  # metadata 필터링 조건 지정
    )
    context = " ".join(docs['documents'][-1:])
    print("\n[LangChain] Connection UUID : ", connection_uuid)
    print("[LangChain] Previous context : ", context)
    print("[LangChain] Converted Text : ", converted_text, "\n")

    textData = f"""
    이전 음성 인식 결과:
    {context}

    현재 음성 인식 결과
    {converted_text}
    """

    # PromptTemplate : 원시 사용자 입력을 더 나은 입력으로 변환
    # OutputParser : 채팅 메시지를 문자열로 변환하는 출력 구문 분석기
    prompt = ChatPromptTemplate.from_template("""
        {textData}
                                              
        아래 조건의 작업을 수행해주세요.
        1. 이전 결과를 고려하여 현재 텍스트를 문법적으로 올바르게 수정해주세요.
        2. 현재 음성 인식 결과에서 잡음이나 인식 오류를 제거해주세요.
        3. 이전 결과와 정말 아무 관련이 없는 내용이 아닌 경우에는 내용을 수정하지 말아주세요
        4. 답변은 한국어로 번역해주세요.
        4-1. 단, 음성인식 결과에 타 언어로 된 전문용어가 들어가 있다면 한국어로 변역하지 말아주세요.
        5. 추가적인 설명 없이 수정된 현재 음성 인식 결과만 출력해주세요.
        6. 문장에 부가 설명을 붙이지 말고 오로지 제공된 현재 음성인식 결과에서만 수정해주세요.
        7. 이전 결과의 내용을 반복하는 형태로 내용을 수정하지 말아주세요
                                              
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

    corrected_text = await asyncio.to_thread(chain1.invoke, {"textData": textData})
    json_result = parse_JSON(corrected_text)

    if json_result:
        result_value = json_result.get('result')
        if result_value:
            print("\n [LangChain]-[" + model_id + "] Result value:", result_value, "\n")
        else:
            return None
    else:
        print("[LangChain]-[" + model_id + "] No 'result' key found in the JSON")
        return None

    # Chroma DB에 데이터 저장
    await asyncio.to_thread(
        vectordb.add_documents,
        documents=[Document(page_content=result_value, metadata={"connection_uuid": connection_uuid})],
        ids=[str(uuid4())],
    )

    return result_value


def delete_data_by_uuid(connection_uuid):
    # connection_uuid에 해당하는 데이터 삭제
    documents = vectordb.get(
        where={"connection_uuid": connection_uuid}  # metadata 필터링 조건 지정
    )

    document_ids = documents["ids"]
    
    # 문서 ID에 해당하는 데이터 삭제
    if document_ids:
        vectordb.delete(ids=document_ids)
        print(f"[LangChain] Data with connection_uuid '{connection_uuid}' has been deleted from ChromaDB.")
    else:
        print(f"[LangChain] No data found with connection_uuid '{connection_uuid}' in ChromaDB.")


######## Problem LangChain ########
async def generate_problems(script, subject, problem_num, problem_types):
    print("\n[LangChain]-[generate_problems] Subject :", subject)
    print("[LangChain]-[generate_problems] Problem_num :", problem_num)
    print("[LangChain]-[generate_problems] Problem Types : ", problem_types, "\n")

    prompt = ChatPromptTemplate.from_template("""
        당신은 대한민국 대학교 {subject} 교수입니다.
        당신은 학생들의 학습 수준을 평가하기 위해서 시험 문제를 출제하는 중입니다.

        {script}

        위 스크립트는 대한민국의 대학교 수준의 {subject}강의 내용인데
        이때 위 스크립트에 기반하여 {problem_num} 개의 문제를 JSON 형식으로 아래 조건에 맞추어서 생성해주세요.

        1. 문제의 Type은 아래와 같이 총 4개만 존재합니다.

        MultipleChoice : 객관식, Option은 네개, 즉 사지선다형
        ShrotAnswer : 단답형
        BlanckQuestion : 빈칸 뚫기 문제
        OXChoice : O X 문제

        2. 주어진 스크립트에서 시험에 나올 수 있는, 중요한 부분에 대한 문제를 생성해주세요.
        3. 추가적인 설명 없이 JSON 결과만 제공해주세요.
        4. 문제 JSON은 아래와 같은 형태여야만 합니다.

            [
                {{
                    "type": "",
                    "direction": "",
                    "options": [
                    "",
                    "",
                    "",
                    ""
                    ],
                    "answer": ""
                }},
                {{
                    // 다음 문제
                }},
                ...
            ]

        아래는 각 JSON의 요소들에 대한 설명입니다. 아래의 설명에 완벽하게 맞추어서 생성해주세요.

        type : 문제 Type 4개 중에 1개

        direction : 문제 질문
        direction : type이 BlanckQuestion인 경우에는 direction에 ___로 빈칸을 뚫어야 한다
        direction : type이 OXChoice인 경우에는 direction이 질문 형태가 아닌 서술 형태로 참 또는 거짓일 수 있어야 한다

        options: MultipleChoice인 경우에만 보기 4개
        options: MultipleChoice이 아닌 다른 Type이면 빈 배열
        options : OXChoice인 경우에도 빈 배열

        answer : 각 문제들에 대한 정답
        answer : MultipleChoice인 경우 options들 중 정답 번호
        answer : ShrotAnswer의 경우 direction에 대한 정답
        answer : BlanckQuestion인 경우 direction에 뚫린 빈칸
        answer : OXChoice인 경우 X인 경우 answer는 0, O인 경우 answer는 1

        5. 이 중에서 {problem_types}에 해당하는 종류의 문제만 생성해주세요
        6. 각 문제의 Type에 맞는 JSON 요소들을 생성해주세요
        7. 항상 모든 문제에 대한 direction과 answer는 꼭 생성해주세요
        8. 문제는 모두 한국어로 생성해주세요
        9. 이를 생성할 때 고민의 시간을 가지고 정확하게 생성해주새요
    """)

    chain = (
        prompt 
        | llm 
        | StrOutputParser()
    )

    problem_result = await asyncio.to_thread(
        chain.invoke, {
               "script" : script,
               "subject" : subject,
               "problem_num" : problem_num,
               "problem_types" : problem_types
        })

    json_result = parse_JSON(problem_result, True)

    if not json_result:
        return None
    
    return json_result