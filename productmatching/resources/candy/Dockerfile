FROM ubuntu:20.04

# Many things in one step to make sure we leave the layer as small as possible
RUN apt-get update \
    && apt-get install -y curl ca-certificates swig git gcc g++ python3-dev python3-pip python3-wheel python3-venv \
       musl-dev make libffi-dev libssl-dev librdkafka-dev libblas-dev liblapack-dev python-dev python-numpy \
       python-setuptools python3-setuptools python3-numpy \
    && curl --silent --show-error https://ca.hdc2.cz/ca.crt -o /usr/local/share/ca-certificates/heureka_ca.crt \
    && update-ca-certificates

RUN pip3 install --no-cache-dir --index-url https://pypi.heu.cz/root/production 'poetry==1.0.0' \
    && poetry config virtualenvs.create false \
    && poetry --version

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN poetry install

COPY . /app

RUN mkdir /tmp/prometheus

ENV PYTHONPATH "${PYTHONPATH}:/app/src/"
ENV prometheus_multiproc_dir /tmp/prometheus

CMD ["python3", "/app/src/candy/main.py"]
