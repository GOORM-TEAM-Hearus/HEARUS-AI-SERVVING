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

        Please perform the following tasks on the value text of the JSON above.
        1. Correct the value text to be grammatically correct.
        2. Remove noise or errors from the value text.
        3. If a sentence ends, please write it in polite (formal) speech.
        4. Please translate the answer into Korean.

        {{
            "result" : "value"
        }}
        Please provide only the improved sentences as the value of the "result" key in JSON format as shown above.
        Please fully maintain the JSON format and answer only with the JSON value without any additional explanations.
        Do not add any explanations to the value text other than the result under any circumstances.
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
        {{
            {script}
        }}

        The above script is university-level lecture content in South Korea.
        We aim to group this script into paragraphs and highlight important key words or sentences.

        [
            "Sentence1",
            "Sentence2",
            ...
        ]
        The script provided will be structured like the above.

        {{
            processedScript : [
                "Paragraph1",
                "Paragraph2",
                ...
                "Last Paragraph"
            ]
        }}
        Please group related sentences into one paragraph and place them within a single string in the processedScript list.
        Be sure to include a comma between the quotation marks that end and start each paragraph.
        Provide the result as JSON data that matches the above format exactly and has only processedScript as the key.

        While adhering to the following conditions, please generate a new processedScript:
        1. If there is any grammatically incorrect content, correct only that.
        2. Surround important words with "**word**" by adding "**" on both sides.
        3. Do not add any data other than the processedScript you received as input.
        4. Keep the JSON format intact and answer only with the JSON value without any additional explanation.
        5. Be meticulous and precise to ensure no strange words are included.
        7. Delete any commas that come after sentence-ending periods (.).
        8. Answer in Korean.
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
async def generate_problems_full(script, subject, problem_num, problem_types):
    total_problems = []
    remaining = problem_num
    # max_attempts = 5  # Limit the number of attempts to prevent infinite loops
    attempts = 0

    while len(total_problems) < problem_num : #and attempts < max_attempts:
        current_num = remaining
        print(f"\n[LangChain]-[generate_problems_full] Generating {current_num} problems (Attempt {attempts + 1})\n")
        generated_problems = await generate_problems(script, subject, current_num, problem_types)

        if not generated_problems:
            print("[LangChain]-[generate_problems_full] No problems generated, stopping.")
            break

        # Ensure generated_problems is a list
        if not isinstance(generated_problems, list):
            generated_problems = [generated_problems]

        total_problems.extend(generated_problems)
        # Remove any duplicates if necessary
        total_problems = list({json.dumps(problem, sort_keys=True): problem for problem in total_problems}.values())
        remaining = problem_num - len(total_problems)
        attempts += 1

    if len(total_problems) < problem_num:
        print(f"[LangChain]-[generate_problems_full] Only {len(total_problems)} problems were generated after {attempts} attempts.")

    # Truncate the list to the desired number of problems
    total_problems = total_problems[:problem_num]

    return total_problems


async def generate_problems(script, subject, problem_num, problem_types):
    print("\n[LangChain]-[generate_problems] Subject :", subject)
    print("[LangChain]-[generate_problems] Problem_num :", problem_num)
    print("[LangChain]-[generate_problems] Problem Types : ", problem_types, "\n")

    prompt = ChatPromptTemplate.from_template("""
        You are a professor of {subject} at a university in South Korea.
        You are currently creating exam questions to assess your students' level of learning.

        {script}

        The above script is content from a university-level {subject} lecture in South Korea.
        Based on the above script, please generate {problem_num} questions in JSON format according to the following conditions.

        1. There are only 2 Types of questions as follows:

        - MultipleChoice: Multiple-choice questions with four options (i.e., four-choice questions)
        - OXChoice: True/False questions (O/X questions)

        2. Please create questions about important parts from the given script that could appear on the exam.
        3. Provide only the JSON result without any additional explanations.
        4. The question JSON must be in the following format:

        ```
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
                // Next question
            }},
            ...
        ]
        ```

        Below is a description of each JSON element. Please generate them perfectly according to the instructions below.

        - type: One of the two question Types
        - direction: The question prompt
        - For OXChoice, the answer to the direction must be either true or false
        - options: Only for MultipleChoice questions, containing four options
        - For OXChoice, an empty array
        - answer: The correct answer for each question
        - For MultipleChoice, the correct option number among the options
        - For OXChoice, use '0' for X (False) and '1' for O (True)

        5. Among these, please generate only the types of questions corresponding to {problem_types}.
        6. For each question, please generate the JSON elements matching its Type.
        7. Always include the direction and answer for all questions.
        8. Please create all questions in Korean.
        9. Please take your time to consider carefully and generate accurately.
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
