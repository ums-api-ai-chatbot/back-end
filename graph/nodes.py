import logging
from typing import Dict, List, Any, Tuple, Optional, TypedDict, Annotated

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

from agent.evaluator import ResponseEvaluator
from config import OPENAI_API_KEY, LLM_MODEL
from prompts.templates import (
    create_retrieval_prompt, 
    create_query_clarification_prompt,
    create_answer_refinement_prompt
)

logger = logging.getLogger(__name__)

# 타입 정의
class ChatHistory(TypedDict):
    messages: List[Dict[str, Any]]

class Question(TypedDict):
    query: str
    chat_history: Optional[List[Dict[str, Any]]]

class QueryClarificationOutput(TypedDict):
    needs_clarification: bool
    clarification_message: Optional[str]
    query: str

class RetrievalOutput(TypedDict):
    context: List[str]
    relevant_docs: List[Document]
    web_search_results: Optional[List[str]]
    needs_web_search: Optional[bool]

class AnswerOutput(TypedDict):
    answer: str
    context: List[str]

class EvaluationOutput(TypedDict):
    hallucination: Dict[str, Any]
    quality: Dict[str, Any]
    needs_refinement: bool

class RefinementOutput(TypedDict):
    final_answer: str

# 노드 함수
def query_clarification(state: Question) -> Dict[str, Any]:
    """
    질문 명확화 노드.
    질문이 명확한지 확인하고, 필요한 경우 명확화를 요청합니다.
    
    Args:
        state: 현재 상태 (질문, 채팅 기록 포함)
        
    Returns:
        Dict[str, Any]: 명확화 결과
    """
    logger.info(f"Clarifying query: {state['query']}")
    
    llm = ChatOpenAI(
        temperature=0,
        model=LLM_MODEL,
        openai_api_key=OPENAI_API_KEY
    )
    
    prompt = create_query_clarification_prompt()
    chain = prompt | llm | StrOutputParser()
    
    result = chain.invoke({"query": state["query"]})
    
    if result.strip() == "CLEAR":
        logger.info("Query is clear, proceeding")
        return {
            "needs_clarification": False,
            "clarification_message": None,
            "query": state["query"]
        }
    else:
        logger.info("Query needs clarification")
        return {
            "needs_clarification": True,
            "clarification_message": result,
            "query": state["query"]
        }

def query_rewriting(state: Question) -> Dict[str, Any]:
    """
    질문 재작성 노드.
    질문을 더 명확하고 검색에 적합하게 재작성합니다.
    
    Args:
        state: 현재 상태 (질문, 채팅 기록 포함)
        
    Returns:
        Dict[str, Any]: 재작성된 질문
    """
    logger.info(f"Rewriting query: {state['query']}")
    
    evaluator = ResponseEvaluator()
    rewritten_query = evaluator.rewrite_query(state["query"])
    
    return {
        "query": rewritten_query,
        "chat_history": state.get("chat_history", [])
    }

def retrieve_context(state: Dict[str, Any], retriever) -> Dict[str, Any]:
    """
    문맥 검색 노드.
    질문과 관련된 문맥을 벡터 저장소에서 검색합니다.
    
    Args:
        state: 현재 상태
        retriever: 문맥 검색기
        
    Returns:
        Dict[str, Any]: 검색된 문맥 정보
    """
    logger.info(f"Retrieving context for query: {state['query']}")
    
    query = state["query"]
    
    try:
        docs = retriever.get_relevant_documents(query)
        logger.info(f"Retrieved {len(docs)} documents")
        
        context_texts = [doc.page_content for doc in docs]
        
        return {
            "context": context_texts,
            "relevant_docs": docs,
            "query": query,
            "chat_history": state.get("chat_history", [])
        }
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return {
            "context": ["문맥 검색 중 오류가 발생했습니다."],
            "relevant_docs": [],
            "query": query,
            "chat_history": state.get("chat_history", [])
        }

def should_search_web(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    웹 검색 필요성을 결정하는 노드.
    벡터 검색 결과가 충분하지 않은 경우 웹 검색을 실행합니다.
    
    Args:
        state: 현재 상태 (문맥 포함)
        
    Returns:
        Dict[str, Any]: 웹 검색 필요 여부
    """
    logger.info("Deciding whether to perform web search")
    
    # 문맥 내용 확인
    context = state.get("context", [])
    
    # 문맥이 없거나 부족한 경우 (예: 2개 미만의 관련 문서)
    if not context or len(context) < 2 or any("문맥 검색 중 오류가 발생했습니다" in text for text in context):
        logger.info("Vector search results insufficient, web search needed")
        return {
            **state,
            "needs_web_search": True
        }
    
    # 문맥 내용이 충분히 명확하고 답변에 적합한지 평가
    is_sufficient = True
    total_context_length = sum(len(text) for text in context)
    
    # 문맥 길이가 너무 짧으면 불충분하다고 판단
    if total_context_length < 300:
        is_sufficient = False
        logger.info(f"Context length ({total_context_length}) is too short, web search needed")
    
    return {
        **state,
        "needs_web_search": not is_sufficient
    }

def web_search(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    인터넷 검색 노드.
    질문에 대한 추가 정보를 웹에서 검색합니다.
    
    Args:
        state: 현재 상태 (질문, 문맥 포함)
        
    Returns:
        Dict[str, Any]: 검색 결과가 추가된 상태
    """
    logger.info(f"Performing web search for query: {state['query']}")
    
    try:
        # DuckDuckGo 검색 래퍼 초기화
        search = DuckDuckGoSearchAPIWrapper()
        
        # 검색 쿼리 생성 (KT-UMS 관련 키워드 추가)
        search_query = f"KT-UMS API {state['query']}"
        
        # 웹 검색 실행
        search_results = search.run(search_query)
        logger.info("Web search completed successfully")
        
        # 검색 결과가 너무 길면 잘라내기
        if len(search_results) > 2000:
            search_results = search_results[:2000] + "..."
            
        # 기존 상태에 검색 결과 추가
        updated_state = state.copy()
        updated_state["web_search_results"] = [search_results]
        
        # 전체 문맥에 웹 검색 결과 추가
        if "context" in updated_state and updated_state["context"]:
            updated_state["context"].append(f"웹 검색 결과: {search_results}")
        
        return updated_state
    except Exception as e:
        logger.error(f"Error during web search: {e}")
        # 오류 발생 시 원래 상태 반환 (웹 검색 결과 없음)
        return state

def generate_answer(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    답변 생성 노드.
    검색된 문맥을 바탕으로 질문에 대한 답변을 생성합니다.
    
    Args:
        state: 현재 상태 (질문, 문맥, 채팅 기록 포함)
        
    Returns:
        Dict[str, Any]: 생성된 답변
    """
    logger.info("Generating answer")
    
    llm = ChatOpenAI(
        temperature=0,
        model=LLM_MODEL,
        openai_api_key=OPENAI_API_KEY
    )
    
    prompt = create_retrieval_prompt()
    
    chain = (
        {
            "query": lambda x: x["query"],
            "context": lambda x: "\n\n".join(x["context"]),
            "chat_history": lambda x: x.get("chat_history", [])
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    try:
        answer = chain.invoke(state)
        logger.info("Answer generated successfully")
        
        return {
            "answer": answer,
            "context": state["context"],
            "query": state["query"]
        }
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return {
            "answer": "죄송합니다. 답변을 생성하는 중 오류가 발생했습니다.",
            "context": state["context"],
            "query": state["query"]
        }

def evaluate_answer(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    답변 평가 노드.
    생성된 답변의 품질과 할루시네이션 여부를 평가합니다.
    
    Args:
        state: 현재 상태 (질문, 답변, 문맥 포함)
        
    Returns:
        Dict[str, Any]: 평가 결과
    """
    logger.info("Evaluating answer")
    
    evaluator = ResponseEvaluator()
    
    query = state["query"]
    answer = state["answer"]
    context = state["context"]
    
    try:
        hallucination_result = evaluator.evaluate_hallucination(query, answer, context)
        quality_result = evaluator.evaluate_quality(query, answer)
        
        # 개선이 필요한지 결정
        needs_refinement = (
            hallucination_result.get("has_hallucination", False) or
            quality_result.get("average_score", 5) < 4.0
        )
        
        logger.info(f"Evaluation complete. Needs refinement: {needs_refinement}")
        
        return {
            "hallucination": hallucination_result,
            "quality": quality_result,
            "needs_refinement": needs_refinement,
            "answer": answer,
            "query": query,
            "context": context
        }
    except Exception as e:
        logger.error(f"Error evaluating answer: {e}")
        return {
            "hallucination": {"error": str(e)},
            "quality": {"error": str(e)},
            "needs_refinement": True,
            "answer": answer,
            "query": query,
            "context": context
        }

def refine_answer(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    답변 개선 노드.
    평가 결과를 바탕으로 답변을 개선합니다.
    
    Args:
        state: 현재 상태 (질문, 답변, 평가 결과 포함)
        
    Returns:
        Dict[str, Any]: 개선된 최종 답변
    """
    logger.info("Refining answer")
    
    llm = ChatOpenAI(
        temperature=0,
        model=LLM_MODEL,
        openai_api_key=OPENAI_API_KEY
    )
    
    prompt = create_answer_refinement_prompt()
    
    # 평가 피드백 작성
    hallucination = state["hallucination"]
    quality = state["quality"]
    
    feedback = []
    
    if "hallucination_score" in hallucination:
        feedback.append(f"할루시네이션 점수: {hallucination['hallucination_score']}/5")
        if "explanation" in hallucination:
            feedback.append(f"할루시네이션 설명: {hallucination['explanation']}")
    
    if "average_score" in quality:
        feedback.append(f"품질 평균 점수: {quality['average_score']}/5")
        if "explanation" in quality:
            feedback.append(f"품질 평가 설명: {quality['explanation']}")
        if "suggestions" in quality:
            feedback.append(f"개선 제안: {quality['suggestions']}")
    
    feedback_text = "\n".join(feedback)
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        refined_answer = chain.invoke({
            "query": state["query"],
            "initial_answer": state["answer"],
            "feedback": feedback_text
        })
        
        logger.info("Answer refined successfully")
        
        return {
            "final_answer": refined_answer
        }
    except Exception as e:
        logger.error(f"Error refining answer: {e}")
        return {
            "final_answer": state["answer"]  # 오류 발생 시 원래 답변 사용
        }