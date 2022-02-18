import json
import numpy
import random
import confluent_kafka
from pprint import pprint

consumer = confluent_kafka.Consumer({
    "bootstrap.servers": "catalogue-confluent-cp-kafka-0.catalogue-confluent-cp-kafka-headless.catalogue-kafka:9092",  # 29092 inside docker
    "group.id": f"mockeris-{random.random()}",
    "default.topic.config": {
        "enable.auto.commit": "false",
        "auto.offset.reset": "earliest",
    },
})

consumer.subscribe([
    "matching-ng-candidate-embedding-cz",
])

print("Consuming")

while True:
    msg = consumer.poll(1)

    if msg is None:
        print('.', end="", flush=True)
        continue

    value = json.loads(msg.value())
    pprint(value)
