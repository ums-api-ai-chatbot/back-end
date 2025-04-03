### 사용예제 코드

```
from langchain.graphs import Graph
from langchain_core.documents import Document
from retrieval_grader import RetrievalGrader
from hallucination_grader import HallucinationGrader
from grade_documents import GradeDocuments
from generate import Generate
from ResultCache import ResultCache
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

## 이 코드는 results-modules-02/를 사용했음.


# LLM 모델 초기화
MODEL_NAME = "gpt-4"
llm = ChatOpenAI(model=MODEL_NAME, temperature=0)

# 평가기 초기화
retrieval_grader = RetrievalGrader(model_name=MODEL_NAME, temperature=0)
hallucination_grader = HallucinationGrader(model_name=MODEL_NAME, temperature=0)

# 캐시 초기화
result_cache = ResultCache()

# LangGraph 그래프 생성
graph = Graph()

# 상태값 초기화
state = {
    "question": "What are the latest advancements in AI research?",
    "documents": [
        Document(
            page_content="Recent studies in AI have focused on generative models like GPT-4.",
            metadata={"source": "AI Journal", "page": 0},
        ),
        Document(
            page_content="Advancements in reinforcement learning have also been significant.",
            metadata={"source": "RL Conference", "page": 1},
        ),
    ],
}

# 노드 정의
def retrieve_node(state):
    """문서 검색 노드"""
    print("==== [RETRIEVE NODE] ====")
    return state  # 이미 초기화된 문서를 사용하므로 그대로 반환


def grade_documents_node(state):
    """문서 관련성 평가 노드"""
    print("==== [GRADE DOCUMENTS NODE] ====")
    grader = GradeDocuments(retrieval_grader)
    return grader.execute(state)


def generate_node(state):
    """답변 생성 노드"""
    print("==== [GENERATE NODE] ====")
    question = state["question"]

    # 캐시 확인
    cached_response = result_cache.get(question)
    if cached_response:
        print("Cache hit: Returning cached response.")
        state["generation"] = cached_response
        return state

    # RAG 체인 초기화
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an assistant generating answers based on relevant documents."),
            ("human", "Relevant documents: \n\n {context} \n\n User question: {question}"),
        ]
    )
    rag_chain = prompt | llm
    generator = Generate(rag_chain)

    # 답변 생성
    result = generator.execute(state)
    result_cache.set(question, result["generation"])  # 캐시에 저장
    return result


def hallucination_check_node(state):
    """환각 여부 평가 노드"""
    print("==== [HALLUCINATION CHECK NODE] ====")
    documents = state["documents"]
    generation = state["generation"]
    hallucination_score = hallucination_grader.evaluate(documents, generation)
    print(f"Hallucination Score: {hallucination_score}")
    state["hallucination_score"] = hallucination_score
    return state


# 그래프 노드 추가
graph.add_node("Retrieve", retrieve_node)
graph.add_node("GradeDocuments", grade_documents_node)
graph.add_node("Generate", generate_node)
graph.add_node("HallucinationCheck", hallucination_check_node)

# 그래프 엣지 정의
graph.add_edge("Retrieve", "GradeDocuments")
graph.add_edge("GradeDocuments", "Generate")
graph.add_edge("Generate", "HallucinationCheck")

# 그래프 실행
final_state = graph.run("Retrieve", state)

# 최종 결과 출력
print("\n==== [FINAL RESULT] ====")
print(f"Question: {final_state['question']}")
print(f"Generated Answer: {final_state['generation']}")
print(f"Hallucination Score: {final_state['hallucination_score']}")
```
