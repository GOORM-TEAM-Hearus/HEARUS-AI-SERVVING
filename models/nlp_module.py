def process_text(text_data):
    for item in text_data.get("unProcessedText", []):
        if "라이트" in item[0]:
            item[1] = "highlight"
        elif "설명" in item[0]:
            item[1] = "comment"
            item[2] = "설명이 포함된 단어는 설명이 추가됩니다"
        elif "프로그래밍" in item[0]:
            item[1] = "comment"
            item[2] = ""
        elif "테스트" in item[0]:
            item[1] = "comment"
            item[2] = ""
        elif "인공지능" in item[0]:
            item[1] = "comment"
            item[2] = ""
        elif "강의" in item[0]:
            item[1] = "comment"
            item[2] = ""
        elif "데이터" in item[0]:
            item[1] = "comment"
            item[2] = ""
        elif "파이썬" in item[0]:
            item[1] = "comment"
            item[2] = ""
        elif "대학" in item[0]:
            item[1] = "comment"
            item[2] = ""

    return text_data
