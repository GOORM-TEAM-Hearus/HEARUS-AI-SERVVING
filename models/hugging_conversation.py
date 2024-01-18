import torch
from transformers import pipeline

class HugPipeline:
    def __init__(self):
        self.pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.bfloat16, device_map="auto")
    
    def create_prompt(self,keyword:str, subject:str):
        # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
        messages = [
            {
                "role": "system",
                "content": f"당신은 {subject}과 교수입니다.",
            },
            {"role": "user", "content": f"{subject}에서 {keyword}이란?"},
        ]
        return self.pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def question(self,keyword:str, subject:str):
        prompt = self.create_prompt(keyword=keyword,subject=subject)
        outputs = self.pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        return outputs[0]["generated_text"]