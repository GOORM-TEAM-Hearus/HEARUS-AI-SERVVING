import torch
import gc
import os
from transformers import pipeline


class HugPipeline:
    def __init__(self):
        self.pipe = pipeline(
            "text-generation",
            model="HuggingFaceH4/zephyr-7b-beta",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def create_prompt(self, keyword: str, subject: str):
        # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
        messages = [
            {
                "role": "system",
                "content": f"당신은 {subject}과 교수입니다.",
            },
            {"role": "user", "content": f"{subject}에서 {keyword}이란?"},
        ]
        return self.pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def question(self, keyword: str, subject: str):
        """
        어떤 전공의 용어를 알고싶을때 씁니다.

        Parameters
        ----------
        keyword : str
        용어.
        subject : str
        전공.

        Examples
        --------

        hp = HugPipeline()
        hp.question(keyword='관성',subject='물리학')
        >> 관성은 물리학적 개체 또는 시스템이 일정 속도로 이동하는 상태에 있을때,.....
        """
        prompt = self.create_prompt(keyword=keyword, subject=subject)
        outputs = self.pipe(
            prompt,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            batch_size=2,
        )
        filtered_output = outputs[0]["generated_text"]
        pos_assistant = filtered_output.find("<|assistant|>")
        return filtered_output[pos_assistant + len("<|assistant|>") + 1 :]


gc.collect()
torch.cuda.empty_cache()
hpl = HugPipeline()
res = hpl.question(keyword="헬륨", subject="화학")
print(res)
