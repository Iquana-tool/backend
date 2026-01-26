FROM python:3.13-slim

# Install uv in the final image
RUN pip install uv

# Set up the environment to use the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Ensure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Install system dependencies required for running the application
RUN apt-get update --allow-unauthenticated && \
    apt-get install -y --no-install-recommends --allow-unauthenticated \
    git \
    openssh-client \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy the rest of the application code
COPY . .

# Setup SSH for private repo access
RUN mkdir -p ~/.ssh && \
    ssh-keyscan github.com >> ~/.ssh/known_hosts && \
    chmod 700 ~/.ssh && \
    chmod 644 ~/.ssh/known_hosts

# Copy SSH key if it exists (for build-time git access)
COPY build_key /tmp/build_key
RUN cp /tmp/build_key ~/.ssh/id_rsa && \
    chmod 600 ~/.ssh/id_rsa

# Sync dependencies using uv
RUN uv sync

# Remove SSH key after installation
RUN rm -f ~/.ssh/id_rsa /tmp/build_key

# Create necessary directories for data and database
RUN mkdir -p data && chmod -R 777 data

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
