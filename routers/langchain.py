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


def parse_JSON(text, is_array=False):
    def extract_json_objects(text):
        json_objects = []
        brace_count = 0
        start_index = None

        for i, char in enumerate(text):
            if char == '{':
                if brace_count == 0:
                    start_index = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_index is not None:
                    json_objects.append(text[start_index:i+1])
                    start_index = None

        return json_objects

    result = []
    json_objects = extract_json_objects(text)
    print("[parse_JSON]-[extract_json_objects]", json_objects)

    for json_str in json_objects:
        try:
            # 줄바꿈과 공백 처리
            json_str = re.sub(r'\s+', ' ', json_str)
            parsed_json = json.loads(json_str)
            print(parsed_json)

            if is_array is False:
                return parsed_json
            result.append(parsed_json)
        except json.JSONDecodeError as e:
            print(f"[LangChain]-[parse_JSON] Error parsing JSON: {str(e)}")
            print(f"[LangChain]-[parse_JSON] Problematic JSON string: {json_str}")

    if not result:
        print("[LangChain]-[parse_JSON] No valid JSON data found in the input text")
        return None

    return result


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
    # print("[LangChain] Previous context : ", context)
    print("[LangChain] Converted Text : ", converted_text, "\n")

    # textData = f"""
    # 이전 음성 인식 결과:
    # {context}

    # 현재 음성 인식 결과
    # {converted_text}
    # """

    textData = f"""
    {{
        "result" : "{converted_text}"
    }}
    """

    # PromptTemplate : 원시 사용자 입력을 더 나은 입력으로 변환
    # OutputParser : 채팅 메시지를 문자열로 변환하는 출력 구문 분석기
    prompt = ChatPromptTemplate.from_template("""
        {textData}

        위 JSON의 value 텍스트에 대해서 아래 조건의 작업을 수행해주세요.
        1. value 텍스트를 문법적으로 올바르게 수정해주세요.
        2. value 텍스트에서 잡음이나 오류를 제거해주세요.
        3. 만약 문장이 끝날 경우 존댓말로 작성해주세요
        4. 답변은 한국어로 번역해주세요.
        
        {{
            "result" : "value"
        }}
        위와 같이 개선된 문장만을 "result" key의 value에 담아 JSON 형태로 제공해주세요
        JSON 형태를 온전하게 지켜서 별도의 설명없이 JSON 값만 답변해주세요
        절대로 결과 외에는 value 텍스트에 설명을 추가하지 마세요
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
    print("[LangChain]-[speech_to_text_modification]", corrected_text)
    json_result = parse_JSON(corrected_text)
    

    if json_result:
        result_value = json_result['result']
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


######## Restructure LangChain ########
async def restructure_script(script):
    print("\n[LangChain]-[restructure_script] script :", script, "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("[LangChain]-[restructure_script] CUDA cache flushed")

    prompt = ChatPromptTemplate.from_template("""
        당신은 대한민국 대학교 교수입니다.

        {script}

        위 스크립트는 대한민국의 대학교 수준의 강의 내용인데
        해당 스크립트를 문단별로 묶고, 중요한 핵심 단어나 문장을 표시하고자 합니다.
	
        [
            "문장1",
            "문장2",
            ...
        ]
        현재 주어지는 스크립트는 위와 같은 구조로 구성되어 있을 것입니다.

        {{
            processedScript : [
                "문단1",
                "문단2",
                ...
                "마지막 문단"
            ]
        }}
        관련있는 문장들을 하나의 문단으로 묶어서 processedScript List의 하나의 String 내에 넣어주세요
        배역의 각 문단이 끝나고 시작하는 " 사이에는 반드시 ,를 넣어주세요
        결과는 위 형태에 반드시 맞추어 유일하게 processedScript를 Key로 가지는 JSON 데이터를 제공해주세요

        이때 아래의 조건을 지키면서 새로운 processedScript를 생성해주세요
        1. 문법적으로 올바르지 않은 내용이 있다면 그것만 수정해주세요
        2. 중요한 단어 양 옆에 **단어** 과 같이 "**"를 붙여주세요
        3. 입력으로 받은 processedScript 이외에 다른 데이터를 추가하지 말아주세요
        4. JSON 형태를 온전하게 지켜서 별도의 설명없이 JSON 값만 답변해주세요
        5. 이상한 단어들이 들어가지 않게 꼼꼼하고 정확하게 생성해주세요
        7. 각 문장의 마침표(.) 다음에 오는 콤마(,)는 삭제해주세요
        8. 한국어로로 답변해주세요
    """)

    chain = (
        prompt 
        | llm 
        | StrOutputParser()
    )

    problem_result = await asyncio.to_thread(
        chain.invoke, {
               "script" : script
        })
    
    print("[LangChain]-[restructure_script] Model Result \n", problem_result)

    json_result = parse_JSON(problem_result, True)

    if not json_result:
        return None
    
    return json_result[0]


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
        ShortAnswer : 단답형
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
        answer : ShortAnswer의 경우 direction에 대한 정답
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
    
    print("[LangChain]-[generate_problems] Model Result \n", problem_result)

    json_result = parse_JSON(problem_result, True)

    if not json_result:
        return None
    
    return json_result
