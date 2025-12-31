# Base Image
FROM python:3.10-slim

# System Dependencies (OpenCV, etc.)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Work Directory
WORKDIR /app

# Copy Requirements
COPY requirements.txt .

# Install Python Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy Code
# We rely on .dockerignore to exclude venv/data
COPY src/ src/
COPY .env .env

# Expose Streamlit Port
EXPOSE 8501

# Run Command
CMD ["python", "-m", "streamlit", "run", "src/app/main.py", "--server.address=0.0.0.0"]
