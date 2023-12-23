import re


def process_text(text):
    # 텍스트에서 특정 패턴이나 키워드를 찾고 처리하는 로직
    # 예: 숫자를 찾아서 '[숫자]'로 대체
    processed_text = re.sub(r"\d+", "[숫자]", text)
    return processed_text
