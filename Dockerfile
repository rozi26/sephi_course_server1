FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y openssl ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Generate Self-Signed SSL Certificate
RUN openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365 -subj "/C=US/ST=Web/L=Cloud/O=Docker/CN=localhost"

EXPOSE 443

CMD ["python", "app.py"]