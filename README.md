![image](https://github.com/user-attachments/assets/9be2766a-7aed-4c24-a1db-16652bb706fd)

## 프로젝트 소개
Hearus는 대학교 교내 청각장애 학우 대필지원 도우미 활동에서 느낀 문제들을 풀어내기 위해 시작되었습니다. </br>
청각장애 학우들이 더 나은 환경에서 학습하고, 비장애 학우들과의 교육적 불평등을 해소할 수 있도록 하기 위해 </br>
인공지능을 활용한 실시간 음성 텍스트 변환과 문제 생성, 하이라이팅 기능을 지닌 서비스입니다.

## MVP Model
![image](https://github.com/user-attachments/assets/6b86e0fc-93fa-4fc4-a77f-1750009f4488)
- 고성능 비동기 처리를 위한 FastAPI 프레임워크 사용
- 실시간 음성 인식 및 자연어 처리 모델 서빙
- API 기반 LLM 및 AI Model 서빙 LangChain 구축

## 주요 기능
![image](https://github.com/user-attachments/assets/56a70ea8-b17e-417a-aeb2-a219a531a3c8)
1. **실시간 음성 인식**: Whisper 모델을 사용하여 고정밀 음성-텍스트 변환 제공

![image](https://github.com/user-attachments/assets/03b429eb-5157-45be-a542-10a368d782a7)
2. **Ollama 하이라이팅, 스크립트 재구조화**: 텍스트 데이터에 대한 고급 분석 및 처리

![image](https://github.com/user-attachments/assets/34f49612-f0cb-4656-bde8-bd356e35924b)
3. **Ollama 문제생성**: LangChain을 활용한 대규모 언어 모델 서비스 구현
</br>
4. **비동기 고성능 처리**: FastAPI의 비동기 기능을 활용한 효율적인 요청 처리

## 기술 스택
| Category | Technology |
|----------|------------|
| Language | Python 3.9+ |
| Framework | FastAPI |
| ASGI Server | Uvicorn |
| AI Models | Whisper, Hugging Face Transformers |
| LLM Integration | LangChain |
| Vector DB | Chroma |
| Development Tools | pip, venv |

## 시작하기
### 필수 요구사항
- Python 3.9 이상
- pip (최신 버전)
- venv (가상 환경 관리)

### 설치 및 실행
1. 레포지토리 clone (이미 완료했다면 skip)
   ```
   git clone https://github.com/TEAM-Hearus/HEARUS-AI-SERVING
   ```

2. 프로젝트 디렉토리로 이동
   ```
   cd HEARUS-AI-SERVING
   ```

3. 가상 환경 생성 및 활성화
   ```
   python -m venv venvs/hearus
   source ./venvs/hearus/Scripts/activate  # Windows
   # source ./venvs/hearus/bin/activate  # macOS/Linux
   ```

4. 의존성 설치
   ```
   pip install -r requirements.txt
   ```

5. Ollama 및 llama3 설치
   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   curl -fsSL https://ollama.com/install.sh | sh

   ollama serve
   ollama pull llama3
   ```

6. 애플리케이션 실행
   ```
   python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## 기여하기
프로젝트에 기여하고 싶으시다면 다음 절차를 따라주세요:
1. 레포지토리를 Fork합니다.
2. 새로운 Branch를 생성합니다 (`git checkout -b feature/NewFeature`).
3. 변경사항을 Commit합니다 (`git commit -m '[FEAT] : ADDED Some Features'`).
4. Branch에 Push합니다 (`git push origin feature/NewFeature`).
5. Pull Request를 생성합니다.

## 라이선스
이 프로젝트는 [MIT License](https://github.com/TEAM-Hearus/.github/blob/main/LICENSE)에 따라 배포됩니다.

## 문의
프로젝트에 대한 문의사항이 있으시다면 [ISSUE](https://github.com/TEAM-Hearus/.github/tree/main/ISSUE_TEMPLATE)를 생성해 주세요.

</br>

---
<p align="center">
  모두의 들을 권리를 위하여 Hearus가 함께하겠습니다
  </br></br>
  <img src="https://img.shields.io/badge/TEAM-Hearus-FF603D?style=for-the-badge" alt="TEAM-Hearus">
</p>
