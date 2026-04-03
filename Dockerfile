FROM python:3.10-slim

WORKDIR /app

# Copy requirement and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose port for our Hackathon UI Dashboard
EXPOSE 8000

# Start a simple web server so HF Spaces can render the beautiful UI Dashboard!
CMD ["python", "-m", "http.server", "8000"]
