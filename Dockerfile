# Use Python image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy everything
COPY . .

# Install dependencies
RUN pip install fastapi chromadb uvicorn

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]