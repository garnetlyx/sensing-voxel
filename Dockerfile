# Sensing Voxel Dockerfile
#
# Multi-stage build for production deployment
# Based on Python 3.12 slim

FROM python:3.12-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata and source needed for editable install
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY config/ ./config/

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[camera]"

# Production stage
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN useradd -m -u 1000 sensingvoxel

# Create required directories with proper permissions
RUN mkdir -p data/calibration data/models data/uploads data/sliced && \
    chown -R sensingvoxel:sensingvoxel /app

# Copy application code
COPY --chown=sensingvoxel:sensingvoxel src/ ./src/
COPY --chown=sensingvoxel:sensingvoxel config/ ./config/

# Switch to non-root user
USER sensingvoxel

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose dashboard port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD python -c "import sensing_voxel; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "sensing_voxel.main"]
