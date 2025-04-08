import os
from pathlib import Path
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 기본 경로 설정
BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "data"
VECTOR_STORE_PATH = Path("./vector_db")  # 상대 경로로 변경

# API 키 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LLM 설정
LLM_MODEL = "gpt-4"
EMBEDDING_MODEL = "text-embedding-3-small"  # 더 작고 효율적인 임베딩 모델로 변경
EVALUATION_MODEL = "gpt-4"  # 할루시네이션 측정용 모델

# 문서 분할 설정
CHUNK_SIZE = 500  # 더 작은 청크 크기로 변경
CHUNK_OVERLAP = 100

# FAISS 벡터 저장소 설정
# VECTOR_STORE_INDEX_NAME은 현재 사용하지 않습니다

# 서버 설정
HOST = "0.0.0.0"
PORT = 8000