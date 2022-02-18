import ujson
import logging
import confluent_kafka


def callback_on_kafka_error(error, msg):
    key = msg.key().decode("utf-8")
    topic = msg.topic().decode("utf-8")
    value = msg.value().decode("utf-8") if msg.value() else None
    logging.error(f"Failed to deliver a message {error} (key {key}, value {value}) to {topic}")


def callback_on_kafka_delivery(error, msg):
    if error is not None:
        callback_on_kafka_error(error, msg)
    else:
        pass


async def kafka_output(remote_services, payload: dict, key: str, topic: str) -> None:
    producer = await remote_services.get("kafka").producer()

    try:
        producer.produce(
            topic=topic,
            key=key,
            value=ujson.dumps(payload),
            on_delivery=callback_on_kafka_delivery
        )

        # Put poll(0) to prevent possible issues with callbacks on_delivery, see https://github.com/confluentinc/confluent-kafka-python/issues/16
        producer.poll(0)
        logging.info(f"Sent {key} to {topic}")
    except confluent_kafka.KafkaException:
        logging.error(f"Can not produce to {topic}")
