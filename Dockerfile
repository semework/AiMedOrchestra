# Dockerfile for AIMedOrchestra agents and orchestrator
FROM python:3.10-slim

WORKDIR /apps

COPY . /apps

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "apps/orchestrator_app.py"]
