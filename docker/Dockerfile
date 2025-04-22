# Dockerfile (for runnable Lambda Container Image - Python 3.12 / Torch 2.6)
# Use the official AWS Lambda base image for Python 3.12
FROM public.ecr.aws/lambda/python:3.12 as base

# Set working directory (standard for Lambda)
WORKDIR /var/task

# Install system dependencies using dnf
RUN dnf update -y && \
  dnf install -y \
  libsndfile \
  gcc-c++ \
  python3-devel \
  && dnf clean all

# Install Python dependencies directly into the environment using requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Optional: Clean pip cache to reduce image size
RUN rm -rf /root/.cache/pip

# Copy the Lambda function code into the working directory
# Make sure 'lambda_function.py' exists in the same directory as the Dockerfile
COPY lambda_function.py .
# If your handler uses other local files/folders, copy them too:
# COPY my_utils/ ./my_utils/

# Set the CMD to your handler (filename.handler_function)
# This tells Lambda what function to invoke within your code.
# Adjust 'lambda_function.lambda_handler' if your file/handler is named differently.
CMD [ "lambda_function.lambda_handler" ]