FROM python:3.11-slim

WORKDIR /app

# Install PyTorch CPU (smaller image — swap for CUDA variant if needed)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install package
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir .

# Default directories for data, models, and output
RUN mkdir -p /app/data /app/models /app/output

ENTRYPOINT ["irrigation-detect"]
CMD ["--help"]
