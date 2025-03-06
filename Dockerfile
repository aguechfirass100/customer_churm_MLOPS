FROM python:3.11-slim

# Install curl
RUN apt-get update && apt-get install -y curl

ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_NO_CACHE_DIR=0 \
    PIP_CACHE_DIR=/root/.cache/pip

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --retries 5 --cache-dir $PIP_CACHE_DIR -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]