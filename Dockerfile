# Python 3.11 이미지를 기반으로 설정
FROM python:3.11

# 작업 디렉토리 설정
WORKDIR /app

# requirements.txt를 먼저 복사
COPY requirements.txt .

# 필요한 패키지 설치 (유저 베이스에 설치)
RUN pip install --no-cache-dir --user -r requirements.txt

# 현재 디렉토리의 모든 파일을 컨테이너의 /app으로 복사
COPY . .

# /data에 패키지 설치를 위한 환경 변수 설정
ENV PYTHONUSERBASE=/data

# PATH에 추가
ENV PATH="/data/bin:$PATH"

# FastAPI 애플리케이션 실행
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
