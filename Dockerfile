# syntax = docker/dockerfile:1.2
FROM  python:3.8.17

RUN --mount=type=secret,id=s1,dst=/etc/secrets/s1 AWS_ACCESS_KEY_ID=$(cat /etc/secrets/s1)
RUN --mount=type=secret,id=s2,dst=/etc/secrets/s2 AWS_SECRET_ACCESS_KEY=$(cat /etc/secrets/s2)
RUN mkdir -p /app
WORKDIR /app/
RUN apt-get update
RUN apt-get install -y git
RUN git clone https://github.com/AahanSingh/UCI-Census-Income-App.git .
RUN pip install -r requirements.txt
# Expose the port that the MLFlow tracking server runs on
EXPOSE 10000
ENTRYPOINT uvicorn main:app --port 10000 --host 0.0.0.0