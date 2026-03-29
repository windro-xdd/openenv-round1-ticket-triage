FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY ticket_triage_env /app/ticket_triage_env
COPY openenv.yaml /app/openenv.yaml
COPY inference.py /app/inference.py
COPY app.py /app/app.py

EXPOSE 8000

CMD ["python", "app.py"]
