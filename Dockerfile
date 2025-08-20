# Stage 1: Build stage

# Use an official Python runtime as the base image
FROM python:3.13-slim AS builder

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install uv
RUN pip install uv

# Copy only the requirements file first
COPY requirements.txt .

# Install dependencies using uv
RUN uv pip install -r requirements.txt

# Stage 2: Final stage
FROM python:3.13-slim

# Install uv in the final image
RUN pip install uv

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Set up the environment to use the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Ensure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Install system dependencies required for running the application
RUN apt-get update --allow-unauthenticated && \
    apt-get install -y --no-install-recommends --allow-unauthenticated \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy the rest of the application code
COPY . .

# Create necessary directories for data and database
RUN chmod -R 777 data

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application using uv
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
