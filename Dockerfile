# Use an official Python 3.12 slim image as the base
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Update package list and install system dependencies for OpenCV
# This is the crucial step that installs libGL.so.1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the Python requirements file
COPY requirements.txt .

# Install Python dependencies
# Ensure gunicorn and flask are in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port Railway will use
EXPOSE 8000

# Set the command to run your Flask/Gunicorn app
# Railway provides the $PORT environment variable
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "app:app"]
