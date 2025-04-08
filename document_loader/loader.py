import logging
import os
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore

from config import (
    DOCS_DIR, 
    VECTOR_STORE_PATH, 
    OPENAI_API_KEY, 
    CHUNK_SIZE, 
    CHUNK_OVERLAP,
    EMBEDDING_MODEL
)
from document_loader.utils import split_documents

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """문서 처리 및 벡터 저장소 생성을 담당하는 클래스"""
    
    def __init__(
        self,
        docs_dir: Path = DOCS_DIR,
        vector_store_path: Path = VECTOR_STORE_PATH,
        embedding_model: str = EMBEDDING_MODEL,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.docs_dir = docs_dir
        self.vector_store_path = vector_store_path
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=OPENAI_API_KEY
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 벡터 저장소 디렉토리 생성
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        
        # 초기화
        self.vector_store = None
    
    def load_documents(self) -> List[Document]:
        """
        지정된 KT-UMS OPEN API 연동규격 문서를 로드합니다.
        
        Returns:
            List[Document]: 로드된 문서 목록
        """
        logger.info(f"Loading documents from {self.docs_dir}")
        
        # 특정 문서 파일 경로 설정
        file_path = self.docs_dir / "KT-UMS OPEN API 연동규격_v1.07.docx"
        
        if not file_path.exists():
            logger.warning(f"Document file not found: {file_path}")
            return []
        
        documents = []
        logger.info(f"Loading document: {file_path.name}")
        try:
            # mode="elements"를 사용하여 문서 구조 요소로 분할
            loader = UnstructuredWordDocumentLoader(str(file_path), mode="elements")
            docs = loader.load()
            documents.extend(docs)
            logger.info(f"Successfully loaded {len(docs)} documents from {file_path.name}")
        except Exception as e:
            logger.error(f"Error loading {file_path.name}: {e}")
        
        logger.info(f"Loaded total of {len(documents)} documents")
        return documents
    
    def process_documents(self) -> Optional[FAISS]:
        """
        문서를 로드하고, 분할하고, 임베딩하여 FAISS 벡터 저장소를 생성합니다.
        
        Returns:
            Optional[FAISS]: 생성된 FAISS 벡터 저장소 또는 None (실패 시)
        """
        logger.info("Processing documents and creating vector store")
        
        # 문서 로드
        documents = self.load_documents()
        if not documents:
            logger.error("No documents to process")
            return None
        
        # 문서 분할
        chunks = split_documents(documents, self.chunk_size, self.chunk_overlap)
        
        # 벡터 저장소 생성
        try:
            # 텍스트 추출
            texts = [doc.page_content for doc in chunks]
            
            # 임베딩 벡터 생성
            embedding_vectors = self.embeddings.embed_documents(texts)
            
            # FAISS 인덱스 생성
            dimension = len(embedding_vectors[0])
            index = faiss.IndexFlatL2(dimension)
            
            # 인메모리 문서 저장소 및 ID 매핑 생성
            docstore = InMemoryDocstore({})
            index_to_docstore_id = {}
            
            # FAISS 벡터 저장소 생성
            vector_store = FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=docstore,
                index_to_docstore_id=index_to_docstore_id
            )
            
            # 문서 추가
            vector_store.add_documents(chunks)
            
            # 벡터 저장소 저장
            self.save_vector_store(vector_store)
            
            self.vector_store = vector_store
            logger.info(f"Successfully created and saved vector store with {len(chunks)} chunks")
            return vector_store
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            return None
    
    def save_vector_store(self, vector_store: FAISS) -> None:
        """
        FAISS 벡터 저장소를 디스크에 저장합니다.
        
        Args:
            vector_store: 저장할 FAISS 벡터 저장소
        """
        save_path = self.vector_store_path
        logger.info(f"Saving vector store to {save_path}")
        
        try:
            vector_store.save_local(str(save_path))
            logger.info(f"Vector store saved successfully with {vector_store.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
    
    def load_vector_store(self) -> Optional[FAISS]:
        """
        디스크에서 FAISS 벡터 저장소를 로드합니다.
        
        Returns:
            Optional[FAISS]: 로드된 FAISS 벡터 저장소 또는 None (실패 시)
        """
        save_path = self.vector_store_path
        index_file = save_path / "index.faiss"
        logger.info(f"Loading vector store from {save_path}")
        
        if not index_file.exists():
            logger.warning("Vector store does not exist. Processing documents...")
            return self.process_documents()
        
        try:
            vector_store = FAISS.load_local(
                folder_path=str(save_path),
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True # 이 설정이 없으면 이전에 만든 FAISS vector db 로드 불가
            )
            
            self.vector_store = vector_store
            logger.info(f"Vector store loaded successfully with {vector_store.index.ntotal} vectors")
            return vector_store
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            logger.warning("Attempting to recreate vector store")
            return self.process_documents()