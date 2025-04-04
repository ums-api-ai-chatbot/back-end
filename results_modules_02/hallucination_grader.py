from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


class GradeHallucinationsModel(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


class HallucinationGrader:
    """생성된 답변의 환각 여부를 평가하는 클래스"""
    def __init__(self, model_name: str, temperature: float = 0):
        """
        HallucinationGrader 초기화
        Args:
            model_name (str): 사용할 LLM 모델 이름 (예: "gpt-4").
            temperature (float): 모델의 응답 다양성을 조정하는 파라미터.
        """
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.structured_llm_grader = self.llm.with_structured_output(GradeHallucinationsModel)

        # 시스템 메시지 정의
        system_message = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

        # 프롬프트 템플릿 생성
        self.hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
            ]
        )

        # 평가기 생성 (프롬프트와 LLM 연결)
        self.grader = self.hallucination_prompt | self.structured_llm_grader

    def evaluate(self, documents: list, generation: str) -> str:
        """
        생성된 답변의 환각 여부를 평가합니다.
        Args:
            documents (list): 생성된 답변을 뒷받침하는 문서 리스트.
            generation (str): 생성된 답변.
        Returns:
            str: "yes" 또는 "no"로 환각 여부를 나타냄.
        """
        result = self.grader.invoke({"documents": documents, "generation": generation})
        return result.binary_score