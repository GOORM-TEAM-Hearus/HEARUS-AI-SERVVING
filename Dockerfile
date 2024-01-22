FROM python:3.9

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install torch==2.1.0+cu118 torchaudio==2.1.0+cu121 torchvision==0.16.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install git+https://github.com/openai/whisper.git

COPY . /app

WORKDIR /app

CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]