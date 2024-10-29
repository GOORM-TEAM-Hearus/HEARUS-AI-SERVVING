![image](https://github.com/user-attachments/assets/9be2766a-7aed-4c24-a1db-16652bb706fd)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

## 프로젝트 소개
Hearus는 대학교 교내 청각장애 학우 대필지원 도우미 활동에서 느낀 문제들을 풀어내기 위해 시작되었습니다. </br>
청각장애 학우들이 더 나은 환경에서 학습하고, 비장애 학우들과의 교육적 불평등을 해소할 수 있도록 하기 위해 </br>
인공지능을 활용한 실시간 음성 텍스트 변환과 문제 생성, 하이라이팅 기능을 지닌 서비스입니다.

## MVP Model
![image](https://github.com/user-attachments/assets/6b86e0fc-93fa-4fc4-a77f-1750009f4488)
- 비동기 처리를 위한 FastAPI 프레임워크 사용
- 실시간 음성 인식 및 자연어 처리 모델 서빙
- API 기반 LLM 및 AI Model 서빙 LangChain 구축

## 주요 기능
![image](https://github.com/user-attachments/assets/56a70ea8-b17e-417a-aeb2-a219a531a3c8)
1. **실시간 음성 인식**: Whisper 모델을 사용하여 음성-텍스트 변환 제공

![image](https://github.com/user-attachments/assets/03b429eb-5157-45be-a542-10a368d782a7)
2. **Ollama 하이라이팅, 스크립트 재구조화**: 텍스트 데이터에 대한 분석 및 처리

![Hearus-OSDC-문제생성 Flow drawio](https://github.com/user-attachments/assets/2e279113-94a1-4110-85fb-0e464e92e12d)
3. **Ollama 문제생성**: LangChain을 활용한 LLM 서비스 구현
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

## 📂 API Document
프로젝트의 API 명세는 아래 링크에서 확인하실 수 있습니다.
</br>
[HEARUS-AI-SERVING/wiki](https://github.com/TEAM-Hearus/HEARUS-AI-SERVING/wiki)

## 📄 라이선스
이 프로젝트는 MIT License 하에 배포됩니다. 
</br>
자세한 내용은 [LICENSE](https://github.com/TEAM-Hearus/HEARUS-AI-SERVING/blob/main/LICENSE) 파일을 참조해주세요.

</br>

---
<p align="center">
  모두의 들을 권리를 위하여 Hearus가 함께하겠습니다
  </br></br>
  <img src="https://img.shields.io/badge/TEAM-Hearus-FF603D?style=for-the-badge" alt="TEAM-Hearus">
</p>
