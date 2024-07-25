FROM python:3.9

WORKDIR /src

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    netcat-openbsd

# Install Ollama
# 빌드 부하 문제로 별도로 인스턴스에 ollama 설치
# RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy application files
COPY main.py requirements.txt start_with_ollama.sh /src/
COPY routers /src/routers
COPY templates /src/templates

# Install application dependencies (excluding torch)
# 빌드 부하 문제로 별도로 인스턴스의 venv에 라이브러리 설치
# 시스템에서 docker-compose volume을 통해 venv내 라이브러리 사용
# RUN grep -v torch requirements.txt > requirements_no_torch.txt && \
#     pip3 install -r requirements_no_torch.txt

# Prepare the start script
RUN sed -i 's/\r$//' /src/start_with_ollama.sh && \
    chmod +x /src/start_with_ollama.sh

CMD ["/bin/bash", "/src/start_with_ollama.sh"]