# Use a slim Python 3.11 image as the base.
FROM python:3.11-slim

# Set working directory inside the container.
WORKDIR /app

# Install system dependencies needed for openc and dlib.
Run apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libopenblas-dev \
    libx11-dev \
    libgtk-3-dev \
    python3-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first - Docker cahces this layer,
# so if requirements don't change, pip install is skipped on rebuild.
COPY requirements.txt .

# Install Python dependencies.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install awscli for S3 interactions.
RUN pip install --no-cache-dir awscli

# Copy the application code.
COPY app/ ./app/

# Copy the run script.
COPY run.sh .
RUN chmod +x run.sh

# Create models directory - will be populated at startup.
RUN mkdir -p models/classifier models/text_classifier

# Expose the port unvicorn will listen on.
EXPOSE 8000

# With this
COPY startup.sh .
RUN chmod +x startup.sh
CMD ["./startup.sh"]


