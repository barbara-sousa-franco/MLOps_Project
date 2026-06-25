# Containerization of the MLOps project — mirrors the kedro-docker standard layout.
# Data is NOT baked in (see .dockerignore); mount it at runtime with -v.
ARG BASE_IMAGE=python:3.13-slim
FROM $BASE_IMAGE AS runtime-environment

# system build tools — a few deps have no prebuilt wheels and compile from source
# (phik <- ydata-profiling needs cmake/C++; twofish <- hopsworks needs gcc)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake && \
    rm -rf /var/lib/apt/lists/*

# install project requirements (pinned, generated from uv.lock)
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install -U pip
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm -f /tmp/requirements.txt

# add a non-root kedro user (security best practice)
ARG KEDRO_UID=999
ARG KEDRO_GID=0
RUN groupadd -f -g ${KEDRO_GID} kedro_group && \
    useradd -m -d /home/kedro_docker -s /bin/bash -g ${KEDRO_GID} -u ${KEDRO_UID} kedro_docker

WORKDIR /home/kedro_docker
USER kedro_docker

FROM runtime-environment

# copy the whole project except what is in .dockerignore
ARG KEDRO_UID=999
ARG KEDRO_GID=0
COPY --chown=${KEDRO_UID}:${KEDRO_GID} . .

EXPOSE 8888

# default: run the full pipeline. Override with: docker run ... --pipeline <name>
CMD ["kedro", "run"]
