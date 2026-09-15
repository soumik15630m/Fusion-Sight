# Fusion service image -- tiny, no GPU. Build context is the repo root so the
# fusion/ package can be copied; keeps the same repo layout as the source.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

COPY fusion/requirements.txt fusion/requirements.txt
RUN pip install -r fusion/requirements.txt

COPY fusion ./fusion

EXPOSE 8100
CMD ["python", "-m", "uvicorn", "fusion.main:app", "--host", "0.0.0.0", "--port", "8100"]
