# Candy

![candy](./candy.gif)

Candidate service for selecting the potential item matching candidates.

## Configure

To connect to the GCP Kafka, copy `secrets.env.example` into `secrets.env` and add credentials.

**Configure extract.**

Add topics for extraction into a constant variable `TOPICS` in `extract.py`

**Configure matching rerun.**

Add topics for rerun matching into a constant variable `TOPIC_NAME` to compere results of the first matchapi decision and
the decision made after a second attempt in the given topic, set `TOPIC_NAME_FINAL` to compare only a final decisions
(if a new `final_decision` will be `no` the script will compare all candidates for the given item) in `matching_rerun.py`.
The `matching_rerun.py` depends on `dump` and `tsv` files collected from `extract.py` script.

## Run! Bitch Run!

To use GCP Kafka instead of local one, replace `docker-compose.kafka_local.yaml` with `docker-compose.kafka_gcp.yaml`.

To use our Kubernetes Kafka instead of local one, replace `docker-compose.kafka_local.yaml` with `docker-compose.kafka_kube.yaml`.

**Prepare kafka for candy (local)**

`docker-compose -f docker-compose.yaml -f docker-compose.build.yaml -f docker-compose.localdev.yaml -f docker-compose.kafka_local.yaml up kafka-topics`

**Run candy.**

`docker-compose -f docker-compose.yaml -f docker-compose.build.yaml -f docker-compose.localdev.yaml -f docker-compose.kafka_local.yaml up candy`

**Run tests.**

`docker-compose -f docker-compose.yaml -f docker-compose.build.yaml -f docker-compose.localdev.yaml -f docker-compose.kafka_local.yaml up tests`

**Run kafdrop (kafka web ui). http://localhost:9000**

`docker-compose -f docker-compose.yaml -f docker-compose.build.yaml -f docker-compose.localdev.yaml -f docker-compose.kafka_local.yaml up kafdrop`

**Run redis-commander (redis web ui). http://localhost:8081**

`docker-compose -f docker-compose.yaml -f docker-compose.build.yaml -f docker-compose.localdev.yaml up redis-commander`

**Export statistics from KUBE Kafka**

Just run `extract_cz` from Gitlab CI Pipeline. **Warning:** Only one instance should run at the time.

**Run matching rerun**

`docker-compose -f docker-compose.yaml -f docker-compose.build.yaml -f docker-compose.localdev.yaml -f docker-compose.kafka_local.yaml up matching_rerun`
