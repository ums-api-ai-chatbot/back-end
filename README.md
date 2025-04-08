# UMS API 챗봇 백엔드 서버

KT-UMS OPEN API 연동규격 문서를 바탕으로 
사용자 질문에 답변하는 챗봇 서비스의 백엔드 서버


## 실행 방법

```bash
venv\Scripts\activate 
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 특이사항

기본적으로 서버 실행 시 init()을 통해 문서를 읽고, 벡터DB로 만들어 vector_db 디렉토리에 저장이 됩니다. (나중에 서버 init시 해당 파일을 읽거나, 없으면 새로 생성하여 저장)

그 뒤 랭그래프 그래프 생성이 되며, 생성이 완료되면 그래프 구조를 chat_graph.png로 만들어 저장합니다. 


## 프로젝트 구조

```
BACK_END/
├── main.py                # FastAPI 메인 애플리케이션
├── config.py              # 환경 설정 및 상수
├── document_loader/       # 문서 처리 관련 컴포넌트트
│   ├── __init__.py
│   └── loader.py          # 문서 로딩 및 벡터 저장 관련
│
├── search/                # 에이전트
│   ├── __init__.py
│   ├── evaluator.py       # 답변 품질 측정
│   └── tools.py           # 인터넷 검색 도구
│
├── graph/                 # 그래프
│   ├── __init__.py
│   ├── nodes.py           # LangGraph 노드 정의
│   ├── edges.py           # LangGraph 엣지 정의
│   └── graph.py           # LangGraph 그래프 및 상태 구성
│
├── prompts/
│   ├── __init__.py
│   ├── system.py          # 프롬프트
│   └── templates.py       # 각 노드별 프롬프트 템플릿
│
├── utils/
│   ├── __init__.py
│   └── helpers.py          # 로깅 및 요청/응답 포맷
│
├── data/
│   └── KT-UMS OPEN API 연동규격_v1.07.docx  # 문서서
│
├── vector_db/
│   ├── index.faiss        # 벡터 index data
│   └── index.pkl          # 문서 deserialized data
│
├── .env                   # 환경 변수 파일
├── requirements.txt       # 필요한 패키지 목록
└── README.md              # 프로젝트 설명
```

## 랭그래프 구조
![랭그래프구조다](chat_graph.png)

## 주요 기술 스택
나중에 채울 것


### 질문 처리

**POST /question**

요청 형식:
```json
{
  "query": "KT-UMS API 연동 방법은 어떻게 되나요?"
}
```

응답 형식:
```json
{
  "answer": "KT-UMS API 연동을 위해서는...",
  "metadata": {
    "original_query": "KT-UMS API 연동 방법은 어떻게 되나요?",
    "rewritten_query": "KT-UMS OPEN API 연동을 위한 구체적인 방법과 필요한 파라미터는 무엇인가요?",
    "processing_time": "2.45s",
    "hallucination_score": 1.5,
    "reliability_score": 8.7,
    "answer_source": "document",
    "internet_search_used": false
  }
}
```

인터넷 검색을 사용한 경우:
```json
{
  "answer": "JSON(JavaScript Object Notation)은 데이터를 교환하기 위한 경량 포맷으로...",
  "metadata": {
    "original_query": "JSON이 뭐야?",
    "rewritten_query": "JSON(JavaScript Object Notation)이란 무엇인가요?",
    "processing_time": "4.32s",
    "hallucination_score": 1.0,
    "reliability_score": 9.2,
    "answer_source": "combined",
    "internet_search_used": true,
    "search_reason": "질문에 일반 기술 개념 키워드가 포함됨",
    "search_query": "JSON 정의 특징 사용 사례",
    "doc_relevance_score": 3.5
  }
}
```



## 라이선스

이 프로젝트의 라이선스는 모르겠으니 KT 라이센스로 하겠습니다. 그런데 이제 DS를 곁들인....
ci/cd 파이프라인 설명입니다!


o CI

o github + git action 사용
github 주소 : https://github.com/ums-api-ai-chatbot/back-end

1. git action trigger 조건 : main 브랜치에 push || pr
2. git action 이 소스코드를 docker image로 말아서 image push 진행
3. image push 하면서 해당 이미지에 태그를 할당
4. 할당된 태그로 deployment.yaml 에 선언된 사용 이미지의 태그 변경

o CD

o kubernetes(k3s) + argocd 사용

서버 주소(kt cloud vm) : 211.254.213.18
argo cd 주소 : http://211.254.213.18:30518
vue 페이지 주소 (front-end) : http://211.254.213.18:32767
fastapi 페이지 주소 (back-end) : http://211.254.213.18:30000

1. k3s로 서버의 전체 pod를 관리합니다.
2. argo cd pod가 git repo의 deployment.yaml을 바라보며 vue ns의 pod 및 fastapi ns의 pod 를 관리 합니다.
3. ci 4번으로 소스코드가 변경되면 argo가 관리하는 pod의 이미지(기존)와 새로운 소스코드의 이미지의 태그가 달라지게 됩니다.
4. argo cd 는 달라진 이미지 태그를 감지하여 out of sync 상태가 됩니다.
5. argo cd 에서 sync를 맟춰주면 관리하는 pod 에서 새로운 이미지를 pull  받아 배포됩니다.
