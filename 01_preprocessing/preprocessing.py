# DocumentLoader - UnstructuredWordDocumentLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
# TextSplitter - RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Embedding - OpenAI Embedding 분야
from langchain_openai import OpenAIEmbeddings
# VectorStore - FAISS
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

# docx 문서에 표로 정리 되어져 있는 비정형 문서
# 비구조화된 워드 문서 로더 인스턴스화
loader = UnstructuredWordDocumentLoader("./data/KT-UMS OPEN API 연동규격_v1.07.docx", mode="elements")

# 문서 로드
docs = loader.load()

# print(len(docs))

# metadata 출력
# print(docs[0].metadata)

# docs 리스트에서 page_content만 추출하여 하나의 문자열로 변환
text = "\n\n".join([doc.page_content for doc in docs])

text_splitter = RecursiveCharacterTextSplitter(
    # 청크 크기를 매우 작게 설정합니다. 예시를 위한 설정입니다.
    chunk_size=500,
    # 청크 간의 중복되는 문자 수를 설정합니다.
    chunk_overlap=100,
    # 문자열 길이를 계산하는 함수를 지정합니다.
    length_function=len,
    # 구분자로 정규식을 사용할지 여부를 설정합니다.
    is_separator_regex=False,
    # "Interface ID"를 기준으로 분할 
    separators=["Interface ID"]
)

chunks = text_splitter.split_text(text)

# print(len(chunks))

# for idx, chunk in enumerate(chunks, start=1):
#     print(idx)
#     print(chunk)

import os
from dotenv import load_dotenv

load_dotenv()  # .env 파일 로드
api_key = os.getenv("OPENAI_API_KEY")  # 환경 변수에서 API 키 가져오기

# OpenAI의 "text-embedding-3-small" 모델을 사용하여 임베딩을 생성합니다.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)

embedding_vectors = embeddings.embed_documents(chunks)

# print(len(embedding_vectors))  # 임베딩 벡터 개수
# print(len(embedding_vectors[0]))  # 각 벡터의 차원 수

dimension_size = len(embedding_vectors[0])

# FAISS 인덱스 & 문서 저장소 생성
index = faiss.IndexFlatL2(dimension_size)  # L2 거리 기반 인덱스
docstore = InMemoryDocstore()  # 문서 저장소
index_to_docstore_id = {} # FAISS 벡터 ID <-> 문서 ID 매핑

db = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id,
)

# 문서 검색 가능하도록 chunks를 Faiss DB에 추가
db.add_texts(chunks)

# print(db.index_to_docstore_id)
# print(f"FAISS에 저장된 벡터 개수: {db.index.ntotal}")  

############ test ##########################

query_text = "SMS API"  # 검색할 문장
similar_docs = db.similarity_search(query_text, k=10)  # 가장 유사한 문서

for i, doc in enumerate(similar_docs):
    print(f" 검색 결과 {i+1}:")
    print(f" 문서 내용: {doc.page_content}")
