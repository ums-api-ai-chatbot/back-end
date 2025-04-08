import logging
from typing import List
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def find_docx_files(docs_dir: Path) -> List[Path]:
    """
    문서 디렉토리에서 모든 .docx 파일을 찾습니다.
    
    Args:
        docs_dir: 문서 디렉토리 경로
        
    Returns:
        List[Path]: 발견된 .docx 파일 경로 목록
    """
    logger.info(f"Searching for .docx files in {docs_dir}")
    
    if not docs_dir.exists():
        logger.warning(f"Directory {docs_dir} does not exist. Creating it.")
        docs_dir.mkdir(parents=True, exist_ok=True)
        return []
    
    docx_files = list(docs_dir.glob("**/*.docx"))
    logger.info(f"Found {len(docx_files)} .docx files")
    
    return docx_files

def split_documents(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """
    문서를 청크로 분할합니다.
    
    Args:
        documents: 분할할 문서 목록
        chunk_size: 각 청크의 최대 크기
        chunk_overlap: 인접 청크 간 겹치는 텍스트의 크기
        
    Returns:
        List[Document]: 분할된 문서 청크 목록
    """
    logger.info(f"Splitting {len(documents)} documents with chunk size {chunk_size} and overlap {chunk_overlap}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        separators=["Interface ID", "\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    logger.info(f"Split documents into {len(chunks)} chunks")
    return chunks