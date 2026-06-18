FROM python:3.12-slim

WORKDIR /app

# deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# app code
COPY wallet_screener.py server.py ./
RUN mkdir -p reports

# Most hosts inject $PORT; default to 8000 locally.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
