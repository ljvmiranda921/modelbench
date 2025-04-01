# Base Stage
FROM python:3.10-slim AS base

ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app
RUN pip install --upgrade pip 
RUN pip install "poetry==2.1.2"
RUN poetry install --without=dev --no-root --no-interaction --no-ansi