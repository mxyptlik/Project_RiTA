# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on
# Render will automatically use the PORT environment variable,
# but it's good practice to document the intended port.
EXPOSE 8001

# Command to run the application using uvicorn
# We'll use the PORT environment variable provided by Render.
# Using the new ChromaDB-based application
# CMD ["uvicorn", "trial_chroma:app", "--host", "0.0.0.0", "--port", "8001"]
