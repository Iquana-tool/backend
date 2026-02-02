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

# Configure git to use token for private repo access (before copying code)
ARG GITHUB_TOKEN
RUN if [ -n "$GITHUB_TOKEN" ]; then \
    cd /tmp && \
    echo "https://${GITHUB_TOKEN}@github.com" > /root/.git-credentials && \
    GIT_CONFIG_GLOBAL=/root/.gitconfig git config --global credential.helper store && \
    GIT_CONFIG_GLOBAL=/root/.gitconfig git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"; \
    fi

# Copy the rest of the application code
COPY . .

# Sync dependencies using uv
RUN uv sync

# Create necessary directories for data and database
RUN mkdir -p data && chmod -R 777 data

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uv", "run", "--upgrade", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
