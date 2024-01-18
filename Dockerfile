FROM python:3.9

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install git+https://github.com/openai/whisper.git

COPY . /app

WORKDIR /app

CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]