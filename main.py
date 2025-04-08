from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from langgraph.graph import Graph
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from typing_extensions import TypedDict, Annotated
from typing import List
from langchain_teddynote.messages import stream_graph, random_uuid
from langchain_core.runnables import RunnableConfig
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

# 캐시 기능 추가
from utils.response_cache import ChatResponseCache

# 템플릿 경로 설정
templates = Jinja2Templates(directory="templates")

from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever

from config import HOST, PORT, DOCS_DIR, VECTOR_STORE_PATH
from document_loader.loader import DocumentProcessor
from graph.edges import create_chat_graph

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("kt_ums_chatbot.log"),
    ],
)

logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="KT-UMS ChatBot API",
    description="E.G DEV 통신AX플랫폼담당 프로젝트",
    version="1.0"
)


# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AppState:
    def __init__(self):
        self.document_processor = None
        self.retriever = None
        self.chat_graph = None
        self.response_cache = ChatResponseCache(max_size=1000, ttl=86400)  # 24시간 TTL로 캐시 생성

app_state = AppState()


# 모델 정의
class QuestionRequest(BaseModel):
    query: str
    chat_history: Optional[List[Dict[str, Any]]] = None

class ChatResponse(BaseModel):
    answer: str
    needs_clarification: bool = False
    clarification_message: Optional[str] = None
    evaluation: Optional[Dict[str, Any]] = None


# 초기화 함수
def init():
    """시스템을 초기화하고 필요한 컴포넌트를 로드합니다."""
    logger.info("Initializing the system...")
    
    try:
        # 디렉토리 생성
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
        
        # 문서 파일 경로 확인
        doc_file_path = DOCS_DIR / "KT-UMS OPEN API 연동규격_v1.07.docx"
        if not doc_file_path.exists():
            logger.warning(f"Document file not found: {doc_file_path}")
            logger.warning("Please place the document file in the data directory")
            return False
        
        # 문서 처리기 초기화
        app_state.document_processor = DocumentProcessor()
        
        # 벡터 저장소 로드 또는 생성
        vector_store = app_state.document_processor.load_vector_store()
        if vector_store is None:
            logger.error("Failed to initialize vector store")
            return False
        
        # 벡터 저장소가 적절히 초기화되었는지 확인
        if not hasattr(vector_store, 'index') or vector_store.index is None or vector_store.index.ntotal == 0:
            logger.error("Vector store was initialized but contains no vectors")
            return False
        
        # 검색기 생성
        app_state.retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # 그래프 생성
        app_state.chat_graph = create_chat_graph(app_state.retriever)
        
        # 그래프가 생성되었는지 확인
        if app_state.chat_graph is None:
            logger.error("Failed to create chat graph")
            return False
        
        # 그래프 시각화 시도
        try:
            # mermaid 코드 생성
            mermaid_code = app_state.chat_graph.get_graph().draw_mermaid()
            
            # 파일에 mermaid 코드 저장
            with open("chat_graph.mmd", "w") as f:
                f.write(mermaid_code)
            logger.info("Graph visualization saved to chat_graph.mmd")
            
            # PNG로 변환 시도
            try:
                import pymermaid
                graph_image = app_state.chat_graph.get_graph().draw_mermaid_png()
                with open("chat_graph.png", "wb") as f:
                    f.write(graph_image)
                logger.info("Graph visualization saved to chat_graph.png")
            except Exception as e:
                logger.warning(f"Failed to create PNG visualization: {e}")
                
        except Exception as e:
            logger.warning(f"Failed to visualize graph: {e}")
        
        logger.info("System initialization completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Unexpected error during initialization: {e}")
        return False

# 엔드포인트
@app.get("/")
async def root():
    return {"message": "KT-UMS API ChatBot is running"}

@app.post("/question", response_model=ChatResponse)
async def ask_question(request: QuestionRequest):
    """질문을 처리하고 응답을 반환합니다."""
    
    logger.info("인입 된 질문 : " + request.query)

    if app_state.chat_graph is None:
        raise HTTPException(status_code=500, detail="초 비 상 !!!!!")
    
    try:
        # 캐시에서 응답 검색
        cached_response = app_state.response_cache.get(request.query, request.chat_history)
        if cached_response:
            logger.info(f"캐시된 응답 반환: {request.query[:30]}...")
            return ChatResponse(**cached_response)
        
        # 캐시 미스 - 그래프 실행
        result = app_state.chat_graph.invoke({
            "query": request.query,
            "chat_history": request.chat_history
        })
        
        # 결과 처리
        if result.get("needs_clarification", False):
            response = ChatResponse(
                answer="",
                needs_clarification=True,
                clarification_message=result.get("clarification_message", "질문을 명확히 해주세요.")
            )
            # 명확화가 필요한 질문은 캐싱하지 않음
            return response
        
        # 최종 답변 결정
        answer = result.get("final_answer") or result.get("answer") or "답변을 생성할 수 없습니다."
        
        # 평가 결과 수집
        evaluation = None
        if "hallucination" in result and "quality" in result:
            evaluation = {
                "hallucination": result["hallucination"],
                "quality": result["quality"]
            }
        
        response = ChatResponse(
            answer=answer,
            evaluation=evaluation
        )
        
        # 재작성된 질문이 있는 경우 함께 저장
        rewritten_query = result.get("rewritten_query", None)
        
        # 응답을 캐시에 저장
        app_state.response_cache.set(
            request.query, 
            response.dict(), 
            request.chat_history,
            rewritten_query=rewritten_query
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

# 캐시 관리 엔드포인트 추가
@app.get("/cache/stats")
async def get_cache_stats():
    """캐시 통계를 반환합니다."""
    return app_state.response_cache.get_stats()

@app.post("/cache/clear")
async def clear_cache():
    """캐시를 비웁니다."""
    app_state.response_cache.clear()
    return {"message": "Cache cleared successfully"}

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 시스템을 초기화합니다."""
    try:
        success = init()
        if success:
            logger.info("System initialized successfully. Chat graph is ready.")
        else:
            logger.error("System initialization failed!")
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")

if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)