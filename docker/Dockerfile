# Use the official AWS Lambda base image for Python 3.9
FROM public.ecr.aws/lambda/python:3.9 AS base

# Set working directory
WORKDIR /var/task

# Install dependencies
RUN yum install -y tar xz curl

# Download and install static ffmpeg binary
RUN curl -L https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz \
  | tar xJ && \
  mv ffmpeg-*-static/ffmpeg /usr/local/bin/ffmpeg && \
  chmod +x /usr/local/bin/ffmpeg

# Install system dependencies
RUN yum update -y && \
  yum install -y \
  libsndfile \
  gcc-c++ \
  python3-devel \
  && yum clean all

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Patch cpuinfo to avoid crashing in Lambda container
RUN pip uninstall -y cpuinfo && pip install py-cpuinfo==9.0.0

# Environment variables to suppress CPU topology inspection issues
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV CPUINFO_NO_WARN=1
ENV CPUINFO_LOG_LEVEL=ERROR

# Optional for MKL-related issues (safe even if not using MKL)
ENV MKL_DEBUG_CPU_TYPE=5

# Copy your Lambda function code
COPY lambda_function.py .

# Set the handler
CMD [ "lambda_function.lambda_handler" ]
