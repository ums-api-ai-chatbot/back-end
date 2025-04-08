from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from prompts.system import KT_UMS_SYSTEM_PROMPT, QUERY_CLARIFICATION_PROMPT

def create_retrieval_prompt() -> ChatPromptTemplate:
    """
    검색 기반 QnA를 위한 프롬프트 템플릿을 생성합니다.
    
    Returns:
        ChatPromptTemplate: 생성된 프롬프트 템플릿
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", KT_UMS_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "질문: {query}"),
        ("system", "다음은 KT-UMS OPEN API 문서에서 검색된 관련 정보입니다:\n\n{context}"),
        ("human", "위 정보를 바탕으로 질문에 답변해주세요.")
    ])
    
    return prompt

def create_query_clarification_prompt() -> ChatPromptTemplate:
    """
    질문 명확화를 위한 프롬프트 템플릿을 생성합니다.
    
    Returns:
        ChatPromptTemplate: 생성된 프롬프트 템플릿
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", QUERY_CLARIFICATION_PROMPT),
        ("human", "{query}")
    ])
    
    return prompt

def create_answer_refinement_prompt() -> ChatPromptTemplate:
    """
    답변 개선을 위한 프롬프트 템플릿을 생성합니다.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 KT-UMS OPEN API에 대한 전문가입니다.
        이전에 생성된 답변을 검토하고 개선해주세요.
        필요한 경우 정확성, 명확성, 구체성을 향상시키세요.
        오류가 있다면 수정하고, 누락된 중요 정보가 있다면 추가하세요.
        답변은 사용자가 KT-UMS OPEN API를 이해하고 사용하는 데 도움이 되어야 합니다."""),

        # 이스케이프 처리된 변수
        ("human", "원래 질문: {query}"),
        ("human", "원래 답변: {initial_answer}"),
        ("human", "평가 피드백: {{hallucination_score}}, {{accuracy_score}}"),
        ("human", "개선된 답변을 제공해주세요.")
    ])
    return prompt