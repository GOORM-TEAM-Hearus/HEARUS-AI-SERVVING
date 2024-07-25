#!/bin/bash

# Wait for Ollama to be ready
while ! nc -z localhost 11434; do   
  sleep 1
done

echo "Ollama is ready. Starting the application..."

exec /venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000