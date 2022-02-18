# All used dependecies and Python packages are in the source image, which is defined here: https://gitlab.heu.cz/catalogue/matching-ng/base-docker-image
# Add more packages at will using either `pip` or `poetry`.
# If package needs to be permanently installed or updated, do so by creating merge request in the referenced repository.

FROM registry.gitlab.heu.cz/catalogue/matching-ng/base-docker-image:latest

WORKDIR /app

ENV PYTHONPATH /app:/app/SourcesPython:/app/resources/candy/src

# to enable prometheus multiprocess logging in candy resource (submodule)
RUN mkdir /tmp/prometheus
ENV prometheus_multiproc_dir /tmp/prometheus

COPY . ./
