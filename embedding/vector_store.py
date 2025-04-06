import os
import logging
import faiss
from typing import List, Optional, Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from config import (
    OPENAI_API_KEY, 
    VECTOR_STORE_PATH, 
    OPENAI_EMBEDDING_MODEL
)

# 로거 설정
logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(
        self, 
        api_key: str = OPENAI_API_KEY,
        embedding_model: str = OPENAI_EMBEDDING_MODEL,
        store_path: str = VECTOR_STORE_PATH
    ):
        """
        벡터 저장소 관리자 초기화
        
        Args:
            api_key          : OpenAI API 키
            embedding_model  : 임베딩 모델명
            store_path       : 벡터 저장소 파일 경로
        """
        self.api_key = api_key
        self.embedding_model = embedding_model
        self.store_path = store_path
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=api_key
        )
        self.vector_store = None
        
    def create_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """
        텍스트 청크의 임베딩 벡터 생성
        
        Args:
            chunks: 임베딩할 텍스트 청크 리스트
            
        Returns:
            List[List[float]]: 임베딩 벡터 리스트
        """
        try:
            logger.info(f"{len(chunks)}개 청크에 대한 임베딩 생성 중...")
            embedding_vectors = self.embeddings.embed_documents(chunks)
            logger.info(f"임베딩 생성 완료: {len(embedding_vectors)}개 벡터 (차원: {len(embedding_vectors[0])})")
            return embedding_vectors
        except Exception as e:
            logger.error(f"임베딩 생성 중 오류 발생: {str(e)}")
            raise
    


    """
        텍스트 청크로부터 FAISS 벡터 저장소 생성
        
        Args:
            chunks: 벡터화할 텍스트 청크 리스트
            
        Returns:
            FAISS: 생성된 벡터 저장소
     """
    def create_vector_store(self, chunks: List[str]) -> FAISS:

        try:
            logger.info(f"{len(chunks)}개 청크에 대한 벡터 저장소 구축 중...")
            
            # 청크가 비어 있는지 확인
            if not chunks:
                logger.warning("벡터화할 텍스트 청크가 없습니다")
                raise ValueError("벡터화할 텍스트 청크가 없습니다")
            
            # 임베딩 생성
            embedding_vectors = self.create_embeddings(chunks)
            dimension_size = len(embedding_vectors[0])
            
            # FAISS 인덱스 & 문서 저장소 생성
            index = faiss.IndexFlatL2(dimension_size)  # L2 거리 기반 인덱스
            docstore = InMemoryDocstore()  # 문서 저장소
            index_to_docstore_id = {}  # FAISS 벡터 ID <-> 문서 ID 매핑
            
            # FAISS 벡터 저장소 생성
            self.vector_store = FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=docstore,
                index_to_docstore_id=index_to_docstore_id,
            )
            
            # 텍스트 청크 추가
            self.vector_store.add_texts(chunks)
            
            logger.info(f"벡터 저장소 생성 완료: {self.vector_store.index.ntotal}개 벡터 인덱싱됨")
            
            # 벡터 저장소 저장
            self.save_vector_store()
            
            return self.vector_store
            
        except Exception as e:
            logger.error(f"벡터 저장소 생성 중 오류 발생: {str(e)}")
            raise
    
    def load_vector_store(self) -> Optional[FAISS]:
        """
        디스크에서 벡터 저장소 로드
        
        Returns:
            Optional[FAISS]: 로드된 벡터 저장소 또는 None
        """
        try:
            # 벡터 저장소 파일 존재 여부 확인
            index_file = f"{self.store_path}.faiss"
            if not os.path.exists(index_file):
                logger.warning(f"벡터 저장소 파일을 찾을 수 없습니다: {index_file}")
                return None
            
            logger.info(f"벡터 저장소 로드 중: {self.store_path}")
            
            # 벡터 저장소 로드
            self.vector_store = FAISS.load_local(
                folder_path=os.path.dirname(self.store_path),
                index_name=os.path.basename(self.store_path),
                embeddings=self.embeddings
            )
            
            logger.info(f"벡터 저장소 로드 완료: {self.vector_store.index.ntotal}개 벡터 인덱싱됨")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"벡터 저장소 로드 중 오류 발생: {str(e)}")
            return None
    
    def save_vector_store(self) -> None:
        """
        벡터 저장소를 디스크에 저장
        """
        if self.vector_store is None:
            logger.warning("저장할 벡터 저장소가 없습니다")
            return
        
        try:
            logger.info(f"벡터 저장소 저장 중: {self.store_path}")
            
            # 디렉토리 생성
            os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
            
            # 저장
            self.vector_store.save_local(
                folder_path=os.path.dirname(self.store_path),
                index_name=os.path.basename(self.store_path)
            )
            
            logger.info(f"벡터 저장소 저장 완료: {self.store_path}")
            
        except Exception as e:
            logger.error(f"벡터 저장소 저장 중 오류 발생: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = 5) -> List[str]:
        """
        쿼리와 유사한 텍스트 청크 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            
        Returns:
            List[str]: 유사한 텍스트 청크 리스트
        """
        if self.vector_store is None:
            logger.error("검색을 위한 벡터 저장소가 초기화되지 않았습니다")
            raise ValueError("검색을 위한 벡터 저장소가 초기화되지 않았습니다")
        
        try:
            logger.info(f"쿼리에 대한 유사 문서 검색 중: '{query}'")
            
            # 유사 문서 검색
            results = self.vector_store.similarity_search(query, k=k)
            
            # 결과에서 텍스트만 추출
            similar_texts = [doc.page_content for doc in results]
            
            logger.info(f"{len(similar_texts)}개의 유사 문서 찾음")
            return similar_texts
            
        except Exception as e:
            logger.error(f"문서 검색 중 오류 발생: {str(e)}")
            raise
            
    def similarity_search_with_metadata(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        쿼리와 유사한 텍스트 청크 검색 (메타데이터 포함)
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            
        Returns:
            List[Dict[str, Any]]: 유사한 텍스트 청크와 메타데이터
        """
        if self.vector_store is None:
            logger.error("검색을 위한 벡터 저장소가 초기화되지 않았습니다")
            raise ValueError("검색을 위한 벡터 저장소가 초기화되지 않았습니다")
        
        try:
            logger.info(f"쿼리에 대한 유사 문서 검색 중 (메타데이터 포함): '{query}'")
            
            # 유사 문서 검색
            results = self.vector_store.similarity_search(query, k=k)
            
            # 결과 형식화
            formatted_results = []
            for i, doc in enumerate(results):
                formatted_results.append({
                    "id": i,
                    "content": doc.page_content,
                    "metadata": doc.metadata if hasattr(doc, "metadata") else {}
                })
            
            logger.info(f"{len(formatted_results)}개의 유사 문서 찾음")
            return formatted_results
            
        except Exception as e:
            logger.error(f"문서 검색 중 오류 발생: {str(e)}")
            raise