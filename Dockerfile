# Dockerfile for a Python application with FastAPI and Uvicorn
# This Dockerfile sets up a lightweight Python environment with necessary dependencies.
FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p models data/raw data/processed

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
