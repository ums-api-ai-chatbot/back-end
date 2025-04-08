import logging
from typing import Dict, Any, List, Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from config import OPENAI_API_KEY, EVALUATION_MODEL

logger = logging.getLogger(__name__)

class ResponseEvaluator:
    """
    LLM 기반 응답 평가기.
    할루시네이션 감지 및 응답 품질 평가를 수행합니다.
    """
    
    def __init__(self, model: str = EVALUATION_MODEL):
        self.llm = ChatOpenAI(
            temperature=0,
            model=model,
            openai_api_key=OPENAI_API_KEY
        )
        
        # 할루시네이션 감지 프롬프트
        self.hallucination_prompt = ChatPromptTemplate.from_template(
            """다음은 사용자의 질문과 AI 시스템의 답변입니다:
            
            사용자 질문: {query}
            
            검색된 문맥: {context}
            
            AI 답변: {response}
            
            위의 답변이 제공된 문맥 정보만을 기반으로 작성되었는지 평가해주세요.
            답변이 제공된 문맥에 없는 정보를 포함하고 있다면 이는 할루시네이션입니다.
            
            1부터 5까지의 척도로 할루시네이션 점수를 매겨주세요:
            1: 할루시네이션 없음 - 답변이 문맥 정보만 사용함
            2: 경미한 할루시네이션 - 약간의 추론이 있으나 대부분 문맥에 기반함
            3: 중간 할루시네이션 - 문맥에 없는 정보가 일부 포함됨
            4: 심각한 할루시네이션 - 상당 부분이 문맥에 없는 정보임
            5: 완전한 할루시네이션 - 답변이 문맥과 거의 무관함
            
            할루시네이션이 있다면, 어떤 부분이 문맥에 없는 정보인지 구체적으로 지적해주세요.
            
            마지막으로, JSON 형식으로 다음 필드를 포함하여 응답해주세요:
            {{
                "hallucination_score": (1-5 사이의 정수),
                "explanation": "할루시네이션에 대한 설명",
                "has_hallucination": (true/false, 점수가 3 이상이면 true)
            }}
            """
        )
        
        # 응답 품질 평가 프롬프트
        self.quality_prompt = ChatPromptTemplate.from_template(
            """다음은 사용자의 질문과 AI 시스템의 답변입니다:
            
            사용자 질문: {query}
            
            AI 답변: {response}
            
            위 답변의 품질을 다음 기준으로 1-5점 척도로 평가해주세요:
            - 정확성: 응답이 질문에 대해 정확한가?
            - 관련성: 응답이 질문과 직접적으로 관련이 있는가?
            - 명확성: 응답이 이해하기 쉬운가?
            - 완전성: 응답이 질문에 완전히 답변하는가?
            
            JSON 형식으로 다음 필드를 포함하여 응답해주세요:
            {{
                "accuracy_score": (1-5 사이의 정수),
                "relevance_score": (1-5 사이의 정수),
                "clarity_score": (1-5 사이의 정수),
                "completeness_score": (1-5 사이의 정수),
                "average_score": (위 점수들의 평균),
                "explanation": "점수에 대한 설명",
                "suggestions": "개선을 위한 제안"
            }}
            """
        )
        
        # 쿼리 재작성 프롬프트
        self.query_rewrite_prompt = ChatPromptTemplate.from_template(
            """다음은 사용자의 원래 질문입니다:
            
            {query}
            
            더 나은 답변을 얻기 위해 이 질문을 더 명확하고 구체적으로 재작성하십시오.
            재작성할 때 다음 사항을 고려하세요:
            - 모호한 부분을 명확히 함
            - 누락된 문맥 정보를 추가함
            - 필요한 경우 질문을 여러 부분으로 나눔
            - KT-UMS OPEN API 관련 용어를 정확히 사용함
            
            전문가는 원래 질문이 이미 충분히 명확하다고 판단하면 원래 질문을 그대로 반환합니다.
            
            재작성된 질문만 반환하고 다른 설명은 추가하지 마세요.
            """
        )
    
    def evaluate_hallucination(
        self, 
        query: str, 
        response: str, 
        context: List[str]
    ) -> Dict[str, Any]:
        """
        응답의 할루시네이션을 평가합니다.
        
        Args:
            query: 사용자 질문
            response: AI 답변
            context: 검색된 문맥 정보
            
        Returns:
            Dict[str, Any]: 할루시네이션 평가 결과
        """
        logger.info("Evaluating hallucination")
        
        # 문맥 텍스트 결합
        context_text = "\n\n".join([doc for doc in context])
            # 입력 데이터 검증
        if not query or not response or not context_text:
            logger.error("Missing required inputs for hallucination evaluation")
            return {
                "hallucination_score": 3,
                "explanation": "필수 입력 데이터가 누락되었습니다.",
                "has_hallucination": True
            }
        # 평가 체인 실행
        chain = self.hallucination_prompt | self.llm | StrOutputParser()
        
        try:
            result = chain.invoke({
                "query": query, 
                "response": response, 
                "context": context_text
            })
            
            # 결과 디버깅
            logger.debug(f"Raw hallucination evaluation result: {result}")
            
            # JSON 추출
            import json
            import re
            json_match = re.search(r'{.*?}', result, re.DOTALL)
            if json_match:
                try:
                    evaluation = json.loads(json_match.group(0))
                    logger.info(f"Hallucination evaluation: {evaluation}")
                    return evaluation
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decoding error: {e}")
                    return {
                        "hallucination_score": 3,
                        "explanation": "JSON 디코딩 중 오류가 발생했습니다.",
                        "has_hallucination": True
                    }
            else:
                logger.warning("Failed to parse hallucination evaluation result")
                return {
                    "hallucination_score": 3,
                    "explanation": "평가 결과를 파싱하는 데 실패했습니다.",
                    "has_hallucination": True
                }
        except Exception as e:
            logger.error(f"Error evaluating hallucination: {e}")
            return {
                "hallucination_score": 3,
                "explanation": f"평가 중 오류 발생: {str(e)}",
                "has_hallucination": True
            }
    
    # def evaluate_quality(self, query: str, response: str, context: List[str]) -> Dict[str, Any]:
    #     """
    #     응답의 할루시네이션을 평가합니다.
    #     """
    #     logger.info("Evaluating hallucination")
        
    #     # 문맥 텍스트 결합
    #     context_text = "\n\n".join(context)
        
    #     # 입력 데이터 검증
    #     if not query or not response or not context_text:
    #         logger.error("Missing required inputs for hallucination evaluation")
    #         return {
    #             "hallucination_score": 3,
    #             "explanation": "필수 입력 데이터가 누락되었습니다.",
    #             "has_hallucination": True
    #         }
        
    #     # 평가 체인 실행
    #     chain = self.hallucination_prompt | self.llm | StrOutputParser()
        
    #     try:
    #         result = chain.invoke({
    #             "query": query,
    #             "response": response,
    #             "context": context_text
    #         })
            
    #         # 결과 디버깅
    #         logger.debug(f"Raw hallucination evaluation result: {result}")
            
    #         # JSON 추출
    #         import json
    #         import re
    #         json_match = re.search(r'{.*?}', result, re.DOTALL)
    #         if json_match:
    #             evaluation = json.loads(json_match.group(0))
    #             logger.info(f"Hallucination evaluation: {evaluation}")
    #             return evaluation
    #         else:
    #             logger.warning("Failed to parse hallucination evaluation result")
    #             return {
    #                 "hallucination_score": 3,
    #                 "explanation": "평가 결과를 파싱하는 데 실패했습니다.",
    #                 "has_hallucination": True
    #             }
    #     except Exception as e:
    #         logger.error(f"Error evaluating hallucination: {e}")
    #         return {
    #             "hallucination_score": 3,
    #             "explanation": f"평가 중 오류 발생: {str(e)}",
    #             "has_hallucination": True
    #         }   
    
    def rewrite_query(self, query: str) -> str:
        """
        질문을 더 명확하게 재작성합니다.
        
        Args:
            query: 원래 사용자 질문
            
        Returns:
            str: 재작성된 질문
        """
        logger.info(f"Rewriting query: {query}")
        
        chain = self.query_rewrite_prompt | self.llm | StrOutputParser()
        
        try:
            rewritten_query = chain.invoke({"query": query})
            logger.info(f"Rewritten query: {rewritten_query}")
            return rewritten_query
        except Exception as e:
            logger.error(f"Error rewriting query: {e}")
            return query  # 오류 발생 시 원래 질문 반환
        
    def evaluate_quality(self, query: str, response: str) -> Dict[str, Any]:
        """
        응답 품질을 평가합니다.
        
        Args:
            query: 사용자 질문
            response: AI 답변
            
        Returns:
            Dict[str, Any]: 품질 평가 결과
        """
        logger.info("Evaluating response quality")
        
        # 평가 체인 실행
        chain = self.quality_prompt | self.llm | StrOutputParser()
        
        try:
            result = chain.invoke({
                "query": query,
                "response": response
            })
            
            # 결과 디버깅
            logger.debug(f"Raw quality evaluation result: {result}")
            
            # JSON 추출
            import json
            import re
            json_match = re.search(r'{.*?}', result, re.DOTALL)
            if json_match:
                evaluation = json.loads(json_match.group(0))
                logger.info(f"Quality evaluation: {evaluation}")
                return evaluation
            else:
                logger.warning("Failed to parse quality evaluation result")
                return {
                    "accuracy_score": 3,
                    "relevance_score": 3,
                    "clarity_score": 3,
                    "completeness_score": 3,
                    "average_score": 3.0,
                    "explanation": "평가 결과를 파싱하는 데 실패했습니다.",
                    "suggestions": "오류를 해결하고 다시 평가해보세요."
                }
        except Exception as e:
            logger.error(f"Error evaluating quality: {e}")
            return {
                "accuracy_score": 3,
                "relevance_score": 3,
                "clarity_score": 3,
                "completeness_score": 3,
                "average_score": 3.0,
                "explanation": f"평가 중 오류 발생: {str(e)}",
                "suggestions": "오류를 해결하고 다시 평가해보세요."
            }