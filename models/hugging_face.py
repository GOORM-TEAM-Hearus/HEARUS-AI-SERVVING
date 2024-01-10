from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import torch

# KoGPT2 모델과 토크나이저 로드
model_name = "skt/kogpt2-base-v2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
    bos_token="</s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    mask_token="<mask>",
)


def create_prompt(keyword, subject):
    return f"대학교 '{subject}' 과목에서 다루는 '{keyword}'의 의미는, "


def generate_text(prompt, max_length):
    # Tokenize Prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate Text
    with torch.no_grad():
        output = model.generate(
            input_ids, max_length=max_length, num_return_sequences=1
        )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text


def add_comment(app, text_data):
    for item in text_data.get("unProcessedText", []):
        if "comment" in item[1]:
            # 전공을 판별하는 것은 추후 개발될 기능
            prompt = create_prompt(item[0], "전공")
            # 현재 Default Prompt는 아래와 같이 설정
            prompt = "'{item[0]}'의 의미는, "

            try:
                input_ids = tokenizer.encode(prompt, return_tensors="pt")
                gen_ids = model.generate(
                    input_ids,
                    max_length=50,
                    repetition_penalty=2.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    use_cache=True,
                )
                generated = tokenizer.decode(gen_ids[0])

                # Strip text from Prompt
                processed_text = generated[len(prompt) :].strip()
                end_idx = processed_text.find("\n")
                if end_idx != -1:
                    processed_text = processed_text[:end_idx].strip()

                app.logger.info(generated)
                item[2] = processed_text
            except Exception as e:
                print(f"Error in generating text: {e}")

    return text_data
