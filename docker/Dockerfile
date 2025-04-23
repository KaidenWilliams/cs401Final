FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
  ffmpeg \
  libsndfile1 \
  && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . /app
WORKDIR /app

RUN pip install gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app", "--log-level", "debug", "--timeout", "120"]
