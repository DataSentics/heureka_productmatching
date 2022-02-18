import os
import ujson
import asyncio
import logging
import confluent_kafka
import time

from aiopyrq.unique_queues import UniqueQueue
from llconfig import Config
from typing import List, Union, Iterable
from buttstrap.remote_services import RemoteServices
from timeit import default_timer as timer

from ..candy import Candy
from candy import probes, metrics
from candy.logic.matchapi import MatchApiManager
from candy.utils import FIBONACCHI


class ClientCandy(Candy):

    def __init__(
            self,
            remote_services: RemoteServices,
            matchapi_manager: MatchApiManager,
            config: Config
    ) -> None:
        """
        Args:
            remote_services:         For communication purposes
            config:                  Config
        """
        super().__init__(remote_services, matchapi_manager, config)
        self.pid = os.getpid()
        self.hostname = os.uname()[1]
        self.items_range_size = config.get('ITEMS_RANGE_SIZE')
        self.item_queue_name = self._get_queue_name(self.topic_redis)
        self.kafka_config = config.get("KAFKA")
        self.unique_queue = None
        self.embedding_quit_queue = None

        logging.debug(f"Will process batches of {self.items_range_size} items.")

    async def init_queue_tasks(self):
        async with self.remote_services.get('redis_offers').context as redis:
            self.unique_queue = UniqueQueue(self.item_queue_name, redis, max_retry_rollback=self.max_retry, max_timeout_in_queue=self.max_minutes)

    async def ready(self) -> bool:
        if self.check_topic_existence:
            if not await self._topic_exists(self.topic_kafka_candidate_embedding):
                raise ValueError("Required topic in Kafka does not exists: " + self.topic_kafka_candidate_embedding)

            if not await self._topic_exists(self.topic_kafka_item_matched):
                raise ValueError(
                    "Required topic in Kafka does not exists: " + self.topic_kafka_item_matched)

            if not await self._topic_exists(self.topic_kafka_item_not_matched):
                raise ValueError("Required topic in Kafka does not exists: " + self.topic_kafka_item_not_matched)

            if not await self._topic_exists(self.topic_kafka_item_unknown):
                raise ValueError(
                    "Required topic in Kafka does not exists: " + self.topic_kafka_item_unknown)

            if not await self._topic_exists(self.topic_kafka_item_no_candidates_found):
                raise ValueError(
                    "Required topic in Kafka does not exists: " + self.topic_kafka_item_no_candidates_found)

        async with self.remote_services.get('redis_offers').context as redis:
            return await redis.execute('PING') == b"PONG"

    async def work(self):
        # we need the loop in process_worker to be at least partially non-blocking
        await asyncio.gather(self.process_worker())

    async def input(self) -> List[bytes]:
        return await self.unique_queue.get_items(self.items_range_size)

    async def process_worker(self):
        if await self.ready():
            probes.ready()

        else:
            logging.error("Candy failed to get immediately ready.")

        await self.update_categories_info(self.supported_categories)

        logging.info("Candy is ready to work.")
        last_probe_time = timer()

        await self.unique_queue.re_enqueue_timeout_items(3600)

        while True:
            probes.live()
            probes.ready()
            metrics.PROBE_TIME_GAP.labels(
                app=metrics.APP_NAME, language=self.language).set(timer() - last_probe_time)
            last_probe_time = timer()

            items_ids = await self.input()

            if not items_ids:
                logging.info("No items. Candy is waiting for christmas...")
                await asyncio.sleep(self.no_items_sleep)
                continue

            with metrics.measure(metrics.LATENCY_METRIC, language=self.language):
                items_ids = [item_id.decode() for item_id in items_ids]

                logging.debug(f"Obtained {items_ids} for matching.")

                try:
                    await self.process_items(items_ids)

                except Exception as e:
                    logging.exception("Cannot process item candidates")
                    await self.rollback(items_ids)
                    metrics.COUNT_EXCEPTION_METRIC.labels('unidentified_process_items_error', self.language).inc()
                    raise e

    async def rollback(self, payload: Iterable) -> None:
        for item in payload:
            await self.rollback_item(item)

    async def rollback_item(self, item: Union[bytes, str, int]) -> None:
        can_rollback = await self.unique_queue.can_rollback_item(item)
        if can_rollback:
            await self.unique_queue.reject_item(item)
        else:
            logging.info(f'Not rolling back item {item}, too much retries')
            await self.ack(item)

    async def ack(self, item: Union[bytes, str, int]) -> None:
        await self.unique_queue.ack_item(item)

    async def get_remains_from_process_queue(self) -> List[str]:
        async with self.remote_services.get('redis_offers').context as redis:
            not_processed_items = await redis.execute('lrange', self.unique_queue.processing_queue_name, 0, -1)

        not_processed_items = [str(item, 'utf-8') for item in not_processed_items]

        return not_processed_items

    async def _redis_output(self, payload: str, queue_key: str) -> None:
        queue_name = self._get_queue_name(queue_key)

        async with self.remote_services.get('redis_monolith_matching').context as redis:
            unique_queue = UniqueQueue(queue_name, redis)
            tries = 0
            while True:
                try:
                    await unique_queue.add_item(payload)
                    break
                except ConnectionResetError as e:
                    if tries > 3:
                        raise e

                    await asyncio.sleep(FIBONACCHI[tries])
                    tries += 1

    def _on_kafka_delivery(self, error, msg):
        if error is not None:
            key = msg.key().decode("utf-8")
            topic = msg.topic().decode("utf-8")
            value = msg.value().decode("utf-8") if msg.value() else None

            metrics.KAFKA_EXCEPTION_METRIC.labels(metrics.APP_NAME, "deliver", topic, self.language).inc()
            logging.error(f"Failed to deliver a message {error} (key {key}, value {value})")

    async def _kafka_output(self, payload: dict, key: str, topic: str) -> None:
        producer = await self.remote_services.get("kafka").producer()
        logging.debug(f"Produce {key} to {topic}.")

        try:
            producer.produce(
                topic=topic,
                key=key,
                value=ujson.dumps(payload),
                on_delivery=self._on_kafka_delivery
            )

            # https://github.com/confluentinc/confluent-kafka-python/issues/16
            producer.poll(0)

        except confluent_kafka.KafkaException as e:
            metrics.KAFKA_EXCEPTION_METRIC.labels(metrics.APP_NAME, "produce", topic, self.language).inc()
            metrics.COUNT_EXCEPTION_METRIC.labels('kafka_produce', self.language).inc()
            logging.error(f"Can not produce to {topic}, because: {e}")

    async def output_item_matched(self, payload: dict) -> None:
        await self._kafka_output(
            payload,
            key=str(payload["item"]["id"]),
            topic=self.topic_kafka_item_matched
        )

        if self.write_result_to_monolith:
            redis_payload = {
                "candidate_id": payload["final_candidate"],
                "final_decision": payload["final_decision"],
                "item_id": str(payload["item"]["id"]),
            }
            await self._redis_output(ujson.dumps(redis_payload), self.topic_monolith_redis)

    async def output_item_not_matched(self, payload: dict) -> None:
        await self._kafka_output(
            payload,
            key=str(payload["item"]["id"]),
            topic=self.topic_kafka_item_not_matched
        )

    async def output_item_unknown(self, payload: dict) -> None:
        await self._kafka_output(
            payload,
            key=str(payload["item"]["id"]),
            topic=self.topic_kafka_item_unknown
        )

    async def output_item_no_candidates_found(self, payload: dict) -> None:
        await self._kafka_output(
            payload,
            key=str(payload["id"]),
            topic=self.topic_kafka_item_no_candidates_found
        )

    async def _topic_exists(self, topic: str) -> bool:
        # Uses list_topics(),
        # because if auto.create.topics.enable is set to true on the broker
        # and an unknown topic is specified it will be created.
        producer = await self.remote_services.get("kafka").producer()
        return topic in producer.list_topics(timeout=10).topics

    def _get_queue_name(self, queue_key: str) -> str:
        return '{}-{}'.format(queue_key, self.language)

    def get_process_queue_name(self) -> str:
        # Probably using only in test
        return self.unique_queue.processing_queue_name

    def _item_redis_id(self, item: dict) -> str:
        """Get item key for redis.

        This is the value that is stored in UniqueQueue.
        """
        return item['id']
