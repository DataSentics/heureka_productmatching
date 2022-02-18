FROM python:3.8-slim-buster

ARG GITLAB_USERNAME
ARG GITLAB_PASSWORD

ENV GITLAB_USERNAME=$GITLAB_USERNAME
ENV GITLAB_PASSWORD=$GITLAB_PASSWORD

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

ENV PYTHONPATH /app/Workflow:/app/resources/candy/src
ENV PYTHONOPTIMIZE 0
ENV LOGLEVEL INFO

RUN apt-get update \
    && apt-get install -y curl ca-certificates git gcc g++ musl-dev make docker.io \
    && curl --silent --show-error https://ca.hdc2.cz/ca.crt -o /usr/local/share/ca-certificates/heureka_ca.crt \
    && update-ca-certificates

RUN pip3 --no-cache-dir install --upgrade pip

RUN pip3 install --no-cache-dir 'poetry==1.0.0' \
    && poetry config virtualenvs.create false \
    && poetry --version

WORKDIR /app

COPY Workflow/pyproject.toml ./
COPY Workflow/poetry.lock ./
RUN poetry install

RUN docker login -u "${GITLAB_USERNAME}" -p "${GITLAB_PASSWORD}" registry.gitlab.heu.cz

COPY . ./

CMD ["python3", "/app/Workflow/main.py"]
