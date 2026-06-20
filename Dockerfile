# Containerization of the MLOps project.
# TODO Dockerfile:
#   - choose a slim base and pin the Python version (project uses >=3.13)
#   - install deps from requirements.txt (pinned versions)
#   - ENTRYPOINT that runs `kedro run` (or serves Prefect)

FROM python:3.13-slim

# Avoid .pyc and force unbuffered stdout/stderr (logs visible in the container)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# TODO: install system dependencies if needed (e.g. build-essential for C libs)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project
COPY . .

# Install the Kedro package (src layout)
RUN pip install --no-cache-dir -e .

# TODO: define the default command.
#   Option A (Kedro):   ENTRYPOINT ["kedro", "run"]
#   Option B (Prefect): ENTRYPOINT ["python", "deployment_prefect.py"]
ENTRYPOINT ["kedro", "run"]
CMD ["--pipeline", "__default__"]
