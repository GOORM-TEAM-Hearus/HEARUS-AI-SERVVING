import os
from uuid import uuid4
from dotenv import load_dotenv
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# .env HUGGINGFACEHUB_API_TOKEN 불러오기
load_dotenv()
huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_token


# 음성 인식 결과를 저장할 Chroma DB 생성
embeddings = HuggingFaceEmbeddings()
vectordb = Chroma(embedding_function=embeddings, persist_directory="./db")


def process_speech_to_text(connection_uuid, converted_text):
    # 이전 음성 인식 결과 검색
    docs = vectordb.similarity_search(converted_text, k=3)
    context = " ".join([doc.page_content for doc in docs])

    # 텍스트 수정을 위한 프롬프트 템플릿
    correction_template = """
    이전 음성 인식 결과:
    {context}

    현재 음성 인식 결과:
    {text}

    이전 결과를 고려하여 현재 텍스트를 문법적으로 올바르게 수정하고, 잡음이나 인식 오류를 제거해주세요.
    문맥을 고려하여 자연스럽게 연결되도록 해주세요.
    """

    correction_prompt = PromptTemplate(
        input_variables=["context", "text"],
        template=correction_template,
    )

    # 텍스트 수정을 위한 LLMChain
    correction_chain = LLMChain(
        llm=HuggingFaceHub(repo_id="gpt2", token=huggingface_token),
        prompt=correction_prompt,
    )

    corrected_text = correction_chain.run({"context": context, "text": converted_text})
    print(corrected_text)

    # 수정된 텍스트를 Chroma DB에 저장
    vectordb.add_texts(
        texts=[corrected_text],
        metadatas=[{"connection_uuid": connection_uuid}],
        ids=[str(uuid4())],
    )


def test():
    # 예시 사용
    connection_uuid = "example_connection_uuid"

    # 첫 번째 음성 인식 결과 처리
    converted_text_1 = "이것은 시장 경제에 대한 설명입니다."
    process_speech_to_text(connection_uuid, converted_text_1)

    # 두 번째 음성 인식 결과 처리
    converted_text_2 = "시장 경제는 가격 아아아아아아아아아 기구를 통해 자원을 배분하는 경제 체제입니다."
    process_speech_to_text(connection_uuid, converted_text_2)

    return "Completed"
