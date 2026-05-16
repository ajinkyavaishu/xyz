FROM python:3.10-slim

WORKDIR /app

# Install system dependencies needed for audio processing and opencv
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Pre-download DeepFace models so the first request doesn't time out
RUN python download_weights.py

# Start Streamlit, binding to the PORT environment variable provided by Render
CMD streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0
