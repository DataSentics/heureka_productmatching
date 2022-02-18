from typing import List, Any

from aiopyrq import UniqueQueue

from .client import ClientCandy

class OcClientCandy(ClientCandy):

    async def _redis_output(self, payload: List[Any], queue_key: str) -> None:
        pass

    def _get_queue_name(self, queue_key: str) -> str:
        return queue_key

    async def get_remains_from_process_queue(self) -> List[str]:
        async with self.remote_services.get('redis_offers').context as redis:
            not_processed_items = await redis.execute('lrange', self.unique_queue.processing_queue_name, 0, -1)

        not_processed_items = [str(item, 'utf-8') for item in not_processed_items]

        return not_processed_items

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

    async def init_queue_tasks(self):
        async with self.remote_services.get('redis_offers').context as redis:
            self.unique_queue = UniqueQueue(self.item_queue_name, redis)

    def _item_redis_id(self, item: dict) -> str:
        """Get item key for redis.

        This is the value that is stored in UniqueQueue.
        """
        return f"product:{item['shop_id']}:{item['id']}"
