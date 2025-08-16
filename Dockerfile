FROM python:3.6-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Expose port
EXPOSE 10000

# Run Streamlit
CMD streamlit run main.py --server.port $PORT --server.address 0.0.0.0
