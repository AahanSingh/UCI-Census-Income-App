# syntax = docker/dockerfile:1.2
FROM  python:3.8.17
ARG s1
ARG s2
ENV AWS_ACCESS_KEY_ID=$s1
ENV AWS_SECRET_ACCESS_KEY=$s2
RUN mkdir -p /app
WORKDIR /app/
RUN apt-get update
RUN apt-get install -y git
RUN git clone https://github.com/AahanSingh/UCI-Census-Income-App.git .
RUN pip install -r requirements.txt
# Expose the port that the MLFlow tracking server runs on
EXPOSE 10000
ENTRYPOINT uvicorn main:app --port 10000 --host 0.0.0.0