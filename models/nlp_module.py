def process_text(text_data):
    for item in text_data.get("unProcessedText", []):
        if "he" in item[0]:
            item[1] = "highlight"
        elif "i" in item[0]:
            item[1] = "comment"
            item[2] = "i가 포함된 단어는 설명이 추가됩니다."

    return text_data
