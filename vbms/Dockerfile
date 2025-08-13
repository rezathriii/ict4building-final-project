# VBMS Platform
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for PyFMI and other packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    gfortran \
    libffi-dev \
    libopenblas-dev \
    liblapack-dev \
    build-essential \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Install main requirements first
RUN pip install --no-cache-dir -r requirements.txt

# Try to install PyFMI separately with fallback
RUN pip install --no-cache-dir pyfmi==2.5 || \
    echo "PyFMI installation failed - will use mock FMU simulator"

# Copy application
COPY . .

# Expose FastAPI port
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
