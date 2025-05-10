# 1. Base image
FROM python:3.13-slim

# 2. Install system dependencies for audio i/o
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Set the working directory
WORKDIR /app

# 4. Intall Python dependencies
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy service code and model weights
COPY backend/app.py ./app.py

# assumes model lives at ./models/best_ast_fma_small.pt
COPY models/ ./models

# copy python package src
COPY src/ ./src

# 6. Expose the service port
EXPOSE 8000

# 7. Run the service
CMD [ "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000" ]
