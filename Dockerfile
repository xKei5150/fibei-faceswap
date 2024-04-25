FROM python:3.9-slim

# Set the working directory
WORKDIR /

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port on which your FastAPI app will run (if applicable)
EXPOSE 8000

# Set the command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]