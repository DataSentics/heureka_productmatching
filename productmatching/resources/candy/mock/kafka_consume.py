import json
import numpy
import random
import confluent_kafka
from pprint import pprint

consumer = confluent_kafka.Consumer({
    "bootstrap.servers": "localhost:9092",  # 29092 inside docker
    "group.id": f"mockeris-{random.random()}",
})

consumer.subscribe([
    "matching-ng-item-matched-cz",
    "matching-ng-item-new-candidate-cz",
    "matching-ng-unknown-match-cz",
])

print("Consuming")

while True:
    msg = consumer.poll(1)
    
    if msg is None:
        continue

    value = json.loads(msg.value())
    pprint(msg.topic())
    pprint(value)
    input()
