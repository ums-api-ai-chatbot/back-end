FROM python:3.11

WORKDIR /back-end

# requirements.txt 파일을 복사합니다.
COPY requirements.txt ./

# dependencies를 설치합니다.
RUN pip install -r requirements.txt

# main.py 파일을 복사합니다.
COPY main.py ./

# templates 파일을 복사합니다.
COPY templates ./templates

# results-dolues-02 파일을 복사합니다.
COPY results-dolues-02 ./results-dolues-02

EXPOSE 8000

# FastAPI 애플리케이션을 실행합니다.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]