# Containerização do projeto MLOps.
# TODO Dockerfile:
#   - escolher base slim e fixar versão de Python (projeto usa >=3.13)
#   - instalar deps a partir do requirements.txt (versões pinadas)
#   - ENTRYPOINT que corre `kedro run` (ou serve o Prefect)

FROM python:3.13-slim

# Evita .pyc e força stdout/stderr sem buffer (logs visíveis no container)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# TODO: instalar dependências de sistema se necessário (ex: build-essential p/ libs C)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o projeto
COPY . .

# Instalar o package Kedro (src layout)
RUN pip install --no-cache-dir -e .

# TODO: definir o comando default.
#   Opção A (Kedro):   ENTRYPOINT ["kedro", "run"]
#   Opção B (Prefect): ENTRYPOINT ["python", "deployment_prefect.py"]
ENTRYPOINT ["kedro", "run"]
CMD ["--pipeline", "__default__"]
