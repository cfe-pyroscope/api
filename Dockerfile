FROM python:3.13-slim

# Set workdir
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    libexpat1-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables (sometimes required)
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY ./app /app

# Make sure Python can find local packages
#ENV PYTHONPATH=

# Run with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
