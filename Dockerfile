FROM python:3.10-slim

WORKDIR /app

EXPOSE 80

COPY ./requirements.txt /app/requirements.txt
RUN apt-get update \
    && apt install --no-install-recommends -y build-essential python3-dev gcc libpcap-dev libsndfile1 wget
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt \
    && rm -rf /root/.cache/pip

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8040"]