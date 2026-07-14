\# 📚 전천후 도서/논문 검증 및 연구 조교 (Academic RAG Verifier)



> "환각(Hallucination) 없는 다중 문서 교차 검증을 위한 오픈소스 연구 조교 플랫폼"



Google Gemini API와 RAG(검색 증강 생성) 기술을 활용하여 일반 서적부터 복잡한 학술 논문(PDF)까지 분석하고, 오직 업로드된 문서에 명시된 사실만으로 답변하여 인용 오류를 100% 차단하는 환각 제로(Zero-Hallucination) 검증 웹 애플리케이션입니다.



\---



✨ 주요 기능 및 기술 명세



| 구분 | 내용 |

|---|---|

| UI/UX | Streamlit 기반 대화형 웹 인터페이스 및 실시간 상태 감지 사이드바 |

| 다중 지원 | TXT, PDF, EPUB, DOCX 등 학술 문서 포맷 다중 업로드 지원 |

| 정밀 파싱 | PyMuPDF(fitz) + 읽기 순서 정렬(`sort=True`)로 2단 편집 논문 및 페이지 번호 완벽 보존 |

| 스마트 청킹 | 800\~1000 토큰(약 3,400자), Overlap 150토큰 — `RecursiveCharacterTextSplitter` 적용 |

| 벡터 DB | ChromaDB (로컬 인메모리 기반, 독립적 데이터 보존) |

| 하이브리드 임베딩 | Google `models/embedding-001` (기본) → 실패 시 로컬 오픈소스 임베딩 자동 폴백 |

| 임베딩 Fallback | `jhgan/ko-sroberta-multitask` (한국어 특화, 768차원) → `all-MiniLM-L6-v2` 자동 전환 |

| 생성 모델 | `gemini-2.5-pro` (고정밀) / `gemini-2.5-flash` (고속) 및 최신 라인업 지원 |

| 생성 Fallback | API 미제공/오류 시 `gemini-2.0-flash` → `gemini-1.5-pro` 등 자동 대체로 무장애 보장 |

| 환각 차단| 검증 전용 시스템 프롬프트 강제 + Temperature 0.0 + 답변 하단 출처(문서명·페이지) 의무 표기 |

| 연구 퀵 액션 | 📌 서지 정보 자동 추출 · 📊 3단 핵심 요약 · 🔍 다중 문서 논리 교차 검증 |



\---



🚀 시작하기 (Quick Start)



1\. 환경 설정 및 의존성 설치

본 프로젝트는 Python 3.10 이상 환경을 권장합니다. 저장소를 클론하고 필요한 라이브러리를 설치하세요.



```bash

git clone https://github.com/seedlink-admin/academic-rag-verifier.git

cd academic-rag-verifier

pip install -r requirements.txt

```



2\. 로컬 서버 실행

```bash

streamlit run app.py

```


3\. 사용 가이드

브라우저(http://localhost:8501) 접속 후 좌측 사이드바에 Google Gemini API Key 입력 (Google AI Studio 무료 발급)

(환경변수 GOOGLE\_API\_KEY 설정 시 자동 인식됩니다)



검증할 문서(PDF/TXT/EPUB/DOCX) 업로드 → 사이드바에서 '✅ 벡터화 완료' 상태 확인

(API Key 없이도 "💻 로컬 임베딩으로 계속하기" 버튼으로 문서 벡터화 테스트가 가능합니다)



상단의 연구용 퀵 액션 버튼을 누르거나 대화창에 논리 검증 질문 입력!

🏗️ 시스템 아키텍처 (Architecture)

\[업로드 파일] → 형식별 파서(PyMuPDF / docx / ebooklib) → 청킹(Recursive Splitter)

&#x20;    ↓

\[임베딩 생성] → Google embedding-001 (실패 시 → 로컬 HF ko-sroberta-multitask 자동 전환)

&#x20;    ↓

\[ChromaDB 저장] → 최초 성공 백엔드로 차원 고정 (벡터 차원 혼용 방지 로직 적용)

&#x20;    ↓

\[질문 \& 검색] → 쿼리 임베딩 → Top-K 유사도 검색 → \[참고 문서] 컨텍스트 구성

&#x20;    ↓

\[Gemini 답변] → 검증 전용 시스템 프롬프트 (Temp=0, 404 시 모델 자동 Fallback)

&#x20;    ↓

\[최종 출력] → 환각 없는 대조 답변 + 정밀 출처 표기 \[출처: 문서명, p.페이지]



📂 프로젝트 구조 (Project Structure)

academic-rag-verifier/

├── app.py               # 메인 Streamlit 애플리케이션 (RAG 핵심 파이프라인 포함)

├── requirements.txt     # 의존성 패키지 목록

├── README.md            # 본 문서

├── LICENSE              # AGPL-3.0 라이선스

└── test/

&#x20;   ├── test\_pipeline.py            # 파싱→청킹→벡터DB 파이프라인 단위 테스트

&#x20;   ├── test\_embedding\_fallback.py  # 임베딩 Fallback 로직 테스트

&#x20;   ├── test\_model\_resolve.py       # 생성 모델 매핑/폴백 로직 테스트

&#x20;   ├── sample\_paper.pdf            # 테스트용 샘플 논문

&#x20;   └── sample\_book.txt             # 테스트용 샘플 텍스트



⚖️ 소유권 및 라이선스 (License \& Contribution)

AGPL-3.0 (GNU Affero General Public License v3.0)



본 프로젝트는 전 세계 연구자들의 학술 공익과 집단지성 활용을 위해 무료 배포 및 오픈소스로 공개됩니다. 개인, 대학원생, 독립 연구자는 자유롭게 복제, 수정 및 사용하실 수 있습니다.



⚠️ 상업적 독점 제한 조항: 본 소스코드를 활용하거나 수정하여 웹 서비스(SaaS) 형태로 유료화 또는 상업적 배포를 하고자 하는 경우, 해당 유료 서비스의 전체 소스코드 역시 AGPL-3.0 라이선스에 의거하여 전 세계에 무료 오픈소스로 공개해야 합니다. (무단 상업적 자산화 및 기업 독점 법적 차단)



소스코드 공개 의무가 면제되는 기업 내부망 전용 폐쇄형 구축(On-Premise) 또는 커스텀 B2B 상업 라이선스가 필요한 기관은 원작자에게 별도로 문의해 주시기 바랍니다.



Author / Project Lead: 이성곤 (SUNGKON LEE)



Copyright: Copyright (c) 2026 이성곤 (SUNGKON LEE). All rights reserved.



Contact \& Feedback: 깃허브 Issue 탭 또는 풀 리퀘스트(PR)를 이용해 주세요.



