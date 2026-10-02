FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

# Review queue by default; override the command for scheduled jobs, e.g.:
#   docker run <image> python pipeline.py run
#   docker run <image> python pipeline.py outcomes
#   docker run <image> python pipeline.py learn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "review_app:app"]
