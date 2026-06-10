FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "run_pipeline.py"]
CMD ["all"]
