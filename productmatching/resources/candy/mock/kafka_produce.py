import json
import numpy
import confluent_kafka

producer = confluent_kafka.Producer({
    "bootstrap.servers": "localhost:9092", # 29092 inside docker
})

# To delete:
# kafka-topics.sh --bootstrap-server localhost:9092 --delete --topic matching-ng-candidate-embedding-cz

msgs = [
    # Apple iPhone 11 64GB
    {"candidate_id": "784155553", "candidate_embedding": ",".join(str(x) for x in numpy.random.rand(100))},
    # GoGEN TVL 19753
    {"candidate_id": "685", "candidate_embedding": ",".join(str(x) for x in numpy.random.rand(100))},
    # Canon DC220
    {"candidate_id": "698", "candidate_embedding": ",".join(str(x) for x in numpy.random.rand(100))},
]

for msg in msgs:
    producer.produce("matching-ng-candidate-embedding-cz", key=msg["candidate_id"], value=json.dumps(msg))

producer.poll(0)
print(producer.flush())