# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy all files to the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the service account file
COPY service-account.json /app/service-account.json

# Set environment variable for Google authentication
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/service-account.json"

# Expose the port the Flask app runs on
EXPOSE 8080

# Command to run the application
CMD ["python", "app.py"]
