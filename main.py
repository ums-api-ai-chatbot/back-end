from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import logging
# 환경 변수 로드
load_dotenv()

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

# 홈 페이지 엔드포인트
@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# API 엔드포인트 정의
@app.post("/items/")
async def create_item(request: QuestionRequest):
    response = llm.invoke(request.question)
    
    print("response", response.content)  
    return {"question": request.question, "answer": response.content}
