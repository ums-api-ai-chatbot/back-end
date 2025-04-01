from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

# 환경 변수 로드
load_dotenv()

# FastAPI 앱 생성
app = FastAPI()

# LLM 객체 생성
llm = ChatOpenAI(
    temperature=0.1,
    model_name="gpt-4o-mini",
)

# 요청 데이터 모델 정의
class QuestionRequest(BaseModel):
    question: str

# FastAPI 엔드포인트 정의
@app.post("/items/")
async def create_item(request: QuestionRequest):
    response = llm.invoke(request.question)
    return {"question": request.question, "answer": response}
