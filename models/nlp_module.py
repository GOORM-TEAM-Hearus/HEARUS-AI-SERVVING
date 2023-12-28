def process_text(text_data):
    for item in text_data.get("unProcessedText", []):
        if "he" in item[0]:
            item[1] = "highlight"
        elif "i" in item[0]:
            item[1] = "comment"

    return text_data
