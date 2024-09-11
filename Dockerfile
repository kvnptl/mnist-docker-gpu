FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

WORKDIR /app

# Set environment variable to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and necessary packages
RUN apt-get update && apt-get install -y \
    software-properties-common \
    python3 \
    python3-dev \
    python3-pip \
    python3-distutils \
    ca-certificates && \
    update-ca-certificates

# Install PyTorch with CUDA 12.1 support
RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install matplotlib

# Copy project files
COPY . /app

# Make the shell script executable
RUN chmod +x /app/run.sh

# Run the training and testing scripts using python3
CMD ["/app/run.sh"]