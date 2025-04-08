import logging
from typing import Dict, Any, TypedDict, List

from langgraph.graph import StateGraph, END
from pydantic import BaseModel
from graph.edges import should_rewrite_query, should_use_internet_search
from graph.nodes import (
    query_understanding_node,
    query_rewrite_node,
    retrieval_node,
    answer_generation_node,
    hallucination_evaluation_node,
    document_relevance_evaluation_node,
    internet_search_evaluation_node,
    generate_search_query_node,
    internet_search_node,
    internet_answer_generation_node,
    combine_answers_node,
    final_answer_node,
)
# 로깅 설정
logger = logging.getLogger(__name__)

# 상태 타입 정의
class ChatbotState(TypedDict):
    query: str  # 사용자의 원래 질문
    original_query: str  # 원래 질문(처리 후)
    rewritten_query: str  # 재작성된 질문
    context: str  # 검색된 문서 컨텍스트
    relevant_docs: List[Dict[str, Any]]  # 검색된 문서 목록
    generated_answer: str  # 생성된 답변
    hallucination_score: float  # 할루시네이션 점수
    reliability_score: float  # 신뢰도 점수
    evaluation_reason: str  # 평가 이유
    improved_answer: str  # 개선된 답변
    doc_relevance_score: float  # 문서 관련성 점수
    doc_evaluation_reason: str  # 문서 평가 이유
    is_kt_ums_doc: bool  # KT-UMS 문서 여부
    need_internet_search: bool  # 인터넷 검색 필요 여부
    search_reason: str  # 인터넷 검색 이유
    search_query: str  # 검색 쿼리
    search_results: str  # 검색 결과
    internet_answer: str  # 인터넷 검색 기반 답변
    combined_answer: str  # 통합 답변
    answer_source: str  # 답변 출처 (document, internet, combined)
    final_answer: str  # 최종 답변

def create_chatbot_graph():
    """KT-UMS API 챗봇 LangGraph 생성"""
    # 그래프 상태 초기화
    graph = StateGraph(ChatbotState)
    
    # 노드 추가
    graph.add_node("query_understanding_node", query_understanding_node)
    graph.add_node("query_rewrite_node", query_rewrite_node)
    graph.add_node("retrieval_node", retrieval_node)
    graph.add_node("answer_generation_node", answer_generation_node)
    graph.add_node("hallucination_evaluation_node", hallucination_evaluation_node)
    graph.add_node("document_relevance_evaluation_node", document_relevance_evaluation_node)
    graph.add_node("internet_search_evaluation_node", internet_search_evaluation_node)
    graph.add_node("generate_search_query_node", generate_search_query_node)
    graph.add_node("internet_search_node", internet_search_node)
    graph.add_node("internet_answer_generation_node", internet_answer_generation_node)
    graph.add_node("combine_answers_node", combine_answers_node)
    graph.add_node("final_answer_node", final_answer_node)
    
    # 1. 질문 이해에서 분기
    graph.add_conditional_edges(
        "query_understanding_node",
        should_rewrite_query,
        {
            "rewrite": "query_rewrite_node",
            "skip_rewrite": "retrieval_node"
        }
    )
    
    # 2. 질문 재작성 후 검색
    graph.add_edge("query_rewrite_node", "retrieval_node")
    
    # 3. 검색 후 답변 생성
    graph.add_edge("retrieval_node", "answer_generation_node")
    
    # 4. 답변 생성 후 할루시네이션 평가
    graph.add_edge("answer_generation_node", "hallucination_evaluation_node")
    
    # 5. 할루시네이션 평가 후 문서 관련성 평가
    graph.add_edge("hallucination_evaluation_node", "document_relevance_evaluation_node")
    
    # 6. 문서 관련성 평가 후 인터넷 검색 필요성 평가
    graph.add_edge("document_relevance_evaluation_node", "internet_search_evaluation_node")
    
    # 7. 인터넷 검색 필요성 평가 후 분기
    graph.add_conditional_edges(
        "internet_search_evaluation_node",
        should_use_internet_search,
        {
            "use_internet": "generate_search_query_node",
            "skip_internet": "final_answer_node"
        }
    )
    
    # 8. 검색 쿼리 생성 후 인터넷 검색
    graph.add_edge("generate_search_query_node", "internet_search_node")
    
    # 9. 인터넷 검색 후 인터넷 기반 답변 생성
    graph.add_edge("internet_search_node", "internet_answer_generation_node")
    
    # 10. 인터넷 기반 답변 생성 후 답변 통합
    graph.add_edge("internet_answer_generation_node", "combine_answers_node")
    
    # 11. 답변 통합 후 최종 답변
    graph.add_edge("combine_answers_node", "final_answer_node")
    
    # 12. 최종 답변 후 종료
    graph.add_edge("final_answer_node", END)
    
    # 시작 노드 설정
    graph.set_entry_point("query_understanding_node")
    
    # 컴파일
    return graph.compile()

# 그래프 인스턴스 생성
kt_ums_chatbot_graph = create_chatbot_graph()

def save_graph_as_png(filepath="./data/langgraph_structure.png"):
    """
    그래프를 PNG 이미지로 저장
    
    Args:
        filepath: 저장할 파일 경로
    
    Returns:
        bool: 저장 성공 여부
    """
    try:
        import os
        from pathlib import Path
        
        # 디렉토리 생성
        Path(os.path.dirname(filepath)).mkdir(parents=True, exist_ok=True)
        
        # PNG 데이터 생성 및 파일로 저장
        png_data = kt_ums_chatbot_graph.get_graph().draw_mermaid_png()
        
        # 바이너리 모드로 파일 저장
        with open(filepath, "wb") as f:
            f.write(png_data)
            
        return True
    except Exception as e:
        logger.error(f"그래프 PNG 저장 중 오류 발생: {e}")
        return False