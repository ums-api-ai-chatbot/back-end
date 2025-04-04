from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


class GradeDocumentsModel(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class RetrievalGrader:
    """문서 검색 결과를 평가하는 클래스"""
    def __init__(self, model_name: str, temperature: float = 0):
        """
        RetrievalGrader 초기화
        Args:
            model_name (str): 사용할 LLM 모델 이름 (예: "gpt-4").
            temperature (float): 모델의 응답 다양성을 조정하는 파라미터.
        """
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.structured_llm_grader = self.llm.with_structured_output(GradeDocumentsModel)

        # 시스템 메시지 정의
        system_message = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

        # 프롬프트 템플릿 생성
        self.grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )

        # 평가기 생성 (프롬프트와 LLM 연결)
        self.grader = self.grade_prompt | self.structured_llm_grader

    def evaluate(self, question: str, document: str) -> str:
        """
        문서와 질문의 관련성을 평가합니다.
        Args:
            question (str): 사용자 질문.
            document (str): 평가할 문서 내용.
        Returns:
            str: "yes" 또는 "no"로 문서의 관련성을 나타냄.
        """
        result = self.grader.invoke({"question": question, "document": document})
        return result.binary_score