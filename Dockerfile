# Use the official Python base image with a specified version
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get clean

# Create and set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install streamlit torch 

RUN pip install transformers

# Start Streamlit and Nginx
CMD ["sh", "-c", "streamlit run main.py"]
