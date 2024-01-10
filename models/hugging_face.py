from transformers import pipeline

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained(
#     "kakaobrain/kogpt",
#     revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
#     bos_token="[BOS]",
#     eos_token="[EOS]",
#     unk_token="[UNK]",
#     pad_token="[PAD]",
#     mask_token="[MASK]",
# )
# model = AutoModelForCausalLM.from_pretrained(
#     "kakaobrain/kogpt",
#     revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
#     pad_token_id=tokenizer.eos_token_id,
#     torch_dtype="auto",
#     low_cpu_mem_usage=True,
# ).to(device="cuda", non_blocking=True)
# _ = model.eval()

model_name = "gpt2"
text_generator = pipeline("text-generation", model=model_name)


def create_prompt(keyword, subject):
    return (
        f"대한민국의 대학교 '{subject}' 과목에서 다루는 '{keyword}'에 대한 자세한 설명을 제공해줘.\n"
        "아래의 조건을 만족하는 답변이어야 해.\n"
        "이 설명은 대학생들이 해당 내용을 이해하는 데 도움이 되어야 해.\n"
        "설명은 요약식으로 100단어 이내로 제공해.\n"
        "꼭 설명이 길 필요는 없어 필요한 내용만 간략하게 제공해줘.\n"
        "설명 이후에 공식이나 코드 등 예시가 필요한 설명의 경우 간략한 예시도 포함해줘.\n"
    )


def add_comment(text_data):
    for item in text_data.get("unProcessedText", []):
        if "comment" in item[1]:
            # subject의 경우 추후 NLP 모델에서 별도로 처리
            prompt = create_prompt(item[0], "전공")

            # with torch.no_grad():
            #     tokens = tokenizer.encode(prompt, return_tensors="pt").to(
            #         device="cuda", non_blocking=True
            #     )
            #     gen_tokens = model.generate(
            #         tokens, do_sample=True, temperature=0.8, max_length=150
            #     )
            #     generated = tokenizer.batch_decode(gen_tokens)[0]

            generated = text_generator(prompt, max_length=1000)[0]["generated_text"]
            end_idx = generated.find("\n", len(prompt))
            item[2] = generated[len(prompt) : end_idx].strip()

    return text_data
