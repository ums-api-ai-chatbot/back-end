import logging
from typing import Dict, List, Any, Annotated, Union, TypedDict, Optional

from langgraph.graph import StateGraph, END

from graph.nodes import (
    query_clarification,
    query_rewriting,
    retrieve_context,
    should_search_web,  # 웹 검색 필요성 결정 노드
    web_search,
    generate_answer,
    evaluate_answer,
    refine_answer
)

logger = logging.getLogger(__name__)

# 그래프의 상태 타입 정의
class GraphState(TypedDict):
    query: str
    chat_history: Optional[List[Dict[str, Any]]]
    needs_clarification: Optional[bool]
    clarification_message: Optional[str]
    context: Optional[List[str]]
    relevant_docs: Optional[List[Any]]
    needs_web_search: Optional[bool]
    web_search_results: Optional[List[str]]
    answer: Optional[str]
    hallucination: Optional[Dict[str, Any]]
    quality: Optional[Dict[str, Any]]
    needs_refinement: Optional[bool]
    final_answer: Optional[str]

def create_chat_graph(retriever) -> StateGraph:
    """
    챗봇 그래프를 생성합니다.
    
    Args:
        retriever: 문맥 검색기
        
    Returns:
        StateGraph: 생성된 그래프
    """
    logger.info("Creating chat graph")
    
    # 노드가 실행되는 방향을 결정하는 라우터
    def should_clarify(state):
        """
        질문 명확화가 필요한지 결정하는 라우터
        """
        if state.get("needs_clarification", False):
            logger.info("질문이 명확하지 않아 종료합니다ㅠㅠ ")
            return "needs_clarification"
        else:
            logger.info("쿼리 재작성 실행행")
            return "continue"
    
    def decide_web_search(state):
        """
        웹 검색이 필요한지 결정하는 라우터
        """
        if state.get("needs_web_search", False):
            logger.info("인터넷 검색 agent ON")
            return "needs_web_search"
        else:
            logger.info("인터넷 검색 안 함.")
            return "skip_web_search"
    
    def should_refine(state):
        """
        답변 개선이 필요한지 결정하는 라우터
        """
        if state.get("답변 개선이 필요합니다.. ㅠㅠㅠㅠ", False):
            logger.info("Routing to refine_answer")
            return "needs_refinement"
        else:
            logger.info("아주 좋은 답변입니다. 개선 필요 없어용")
            return "final_answer"
    
    # 그래프 생성
    builder = StateGraph(GraphState)
    
    # 노드 추가
    builder.add_node("query_clarification", query_clarification)
    builder.add_node("query_rewriting", query_rewriting)
    builder.add_node("retrieve_context", lambda state: retrieve_context(state, retriever))
    builder.add_node("should_search_web", should_search_web) 
    builder.add_node("web_search", web_search)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("evaluate_answer", evaluate_answer)
    builder.add_node("refine_answer", refine_answer)
    
    # 엣지 설정 (조건부 라우팅)
    builder.add_conditional_edges(
        "query_clarification",
        should_clarify,
        {
            "needs_clarification": END,
            "continue": "query_rewriting"
        }
    )
    
    # 기본 순차적 엣지 설정
    builder.add_edge("query_rewriting", "retrieve_context")
    
    # 문맥 검색 후 웹 검색 여부 결정
    builder.add_edge("retrieve_context", "should_search_web")
    
    # 웹 검색 여부에 따른 조건부 라우팅
    builder.add_conditional_edges(
        "should_search_web",
        decide_web_search,
        {
            "needs_web_search": "web_search",
            "skip_web_search": "generate_answer"
        }
    )
    
    # 웹 검색 후 답변 생성으로 진행
    builder.add_edge("web_search", "generate_answer")
    
    # 답변 생성 후 평가
    builder.add_edge("generate_answer", "evaluate_answer")
    
    # 답변 평가 후 조건부 라우팅
    builder.add_conditional_edges(
        "evaluate_answer",
        should_refine,
        {
            "needs_refinement": "refine_answer",
            "final_answer": END
        }
    )
    
    # 답변 개선 후 종료
    builder.add_edge("refine_answer", END)
    
    # 시작 노드 설정
    builder.set_entry_point("query_clarification")
    
    # 그래프 컴파일
    graph = builder.compile()
    
    logger.info("Chat graph created successfully")
    return graph