from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from langgraph.graph import Graph
from langchain_core.documents import Document
from results_modules_02 import RetrievalGrader, HallucinationGrader, ResultCache, Generate
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

import logging
# 환경 변수 로드
load_dotenv()
MODEL_NAME="gpt-4o-mini"
# FastAPI 앱 생성
app = FastAPI()

# 템플릿 경로 설정
templates = Jinja2Templates(directory="templates")

# LLM 객체 생성
llm = ChatOpenAI(
    temperature=0.1,
    model_name="gpt-4o-mini",
)

# 요청 데이터 모델 정의
class QuestionRequest(BaseModel):
    question: str
# 그래프의 상태 정의
class GraphState(TypedDict):
    """
    그래프의 상태를 나타내는 데이터 모델

    Attributes:
        question: 질문
        generation: LLM 생성된 답변
        documents: 도큐먼트 리스트
    """

    question: Annotated[str, "User question"]
    generation: Annotated[str, "LLM generated answer"]
    documents: Annotated[List[str], "List of documents"]
# 평가기 초기화
retrieval_grader = RetrievalGrader(model_name=MODEL_NAME, temperature=0)
hallucination_grader = HallucinationGrader(model_name=MODEL_NAME, temperature=0)

# 캐시 초기화
result_cache = ResultCache()

# LangGraph 그래프 생성
graph = StateGraph(GraphState)


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
    "generation": None,
    "hallucination_score": None,
}

# 노드 정의
def retrieve_node(state):
    """문서 검색 노드"""
    print("==== [RETRIEVE NODE] ====")
    print(state["question"])
    return state  # 이미 초기화된 문서를 사용하므로 그대로 반환


def grade_documents_node(state):
    """문서 관련성 평가 노드"""
    print("==== [GRADE DOCUMENTS NODE] ====")
    # Use the existing retrieval_grader instance
    state["graded_documents"] = retrieval_grader.evaluate(state["question"], state["documents"])
    print(f"Graded documents: {state['graded_documents']}")
    return state

def generate_node(state):
    """답변 생성 노드"""
    print("==== [GENERATE NODE] ====")
    question = state["question"]

    # Generate response
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an assistant generating answers based on relevant documents. If you don't know the answer, say 'sorry, I don't know'."),
            ("human", "Relevant documents: \n\n {context} \n\n User question: {question}"),
        ]
    )
    rag_chain = prompt | llm
    generator = Generate(rag_chain)
    result = generator.execute(state)

    # Extract the content of the AIMessage object
    state["generation"] = result["generation"].content if hasattr(result["generation"], "content") else str(result["generation"])

    # Cache the result
    result_cache.set(question, state["generation"])

    print(f"Generated response: {state['generation']}")
    return state  # Return the updated state



def hallucination_check_node(state):
    """환각 여부 평가 노드"""
    print("==== [HALLUCINATION CHECK NODE] ====")
    documents = state["documents"]
    generation = state["generation"]
    hallucination_score = hallucination_grader.evaluate(documents, generation)
    print(f"Hallucination Score: {hallucination_score}")
    state["hallucination_score"] = hallucination_score
    return state  # Return the updated state


@app.on_event("startup")
def startup_event():
    print("어플리케이션 실행 후 실행됨")
    # 그래프 상태 초기화

    # 그래프 노드 추가
    graph.add_node("Retrieve", retrieve_node)
    graph.add_node("RetrievalGrader", grade_documents_node)
    graph.add_node("Generate", generate_node)
    graph.add_node("HallucinationCheck", hallucination_check_node)

    # 그래프 엣지 정의
    graph.add_edge(START, "Retrieve")
    graph.add_edge("Retrieve", "RetrievalGrader")
    graph.add_edge("RetrievalGrader", "Generate")
    graph.add_edge("Generate", END)
    


# 홈 페이지 엔드포인트
@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# API 엔드포인트 정의
@app.post("/items/")
async def create_item(request: QuestionRequest):
    response = llm.invoke(request.question)
    
    print("response", response.content)  
    return {"question": request.question, "generate": response.content}


@app.post("/question/")
async def process_question(request: QuestionRequest):
    app = graph.compile(checkpointer=MemorySaver())
    config = RunnableConfig(recursion_limit=20, configurable={"thread_id": random_uuid()})
    state["question"] = request.question

    # Execute the graph
    try:
        stream_graph(app, state, config, ["agent", "generate_node"])
    except Exception as e:
        print(f"Error during graph execution: {e}")
        raise e
    print("State after graph execution:", state)

    return {"question": state["question"], "generate": state['generation'], "documents": state["documents"], "hallucination_score": state["hallucination_score"]}