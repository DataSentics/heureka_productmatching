import os
from collections import defaultdict
from typing import List, Tuple
from unittest.mock import patch, MagicMock, call

from aiopyrq.unique_queues import UniqueQueue

import confluent_kafka
import confluent_kafka.admin
import pytest
import ujson

from llconfig import Config

from buttstrap.remote_services import RemoteServices
from matching_common.clients.provider import FAISS_CANDIDATES_SOURCE, ELASTIC_CANDIDATES_SOURCE

from candy.config import init_config
from candy.logic.providers import Candidate
from candy.logic.implementation.client import ClientCandy

ITEM_QUEUE_KEY = 'test-items'


@pytest.mark.asyncio
async def init_test(item_ids_to_match: List[int], item_queue_name: str) -> Tuple[RemoteServices, Config]:
    config = init_config()

    config['WRITE_RESULTS_TO_MONOLITH'] = True
    config["KAFKA"]["bootstrap.servers"] = "kafka:29092"
    config["KAFKA"]["consumer"]["group.id"] = "candy_test"
    config["KAFKA"]["consumer"]["enable.auto.commit"] = True
    config["KAFKA"]["consumer"]["default.topic.config"] = {
        "auto.offset.reset": "earliest"
    }
    config["SUPPORTED_CATEGORIES"] = "2,45,123,234,342"

    config["PRIORITIZE_STATUS"] = True

    remote_services = RemoteServices(
        config,
        kafka=['kafka'],
        redis=['redis_offers', 'redis_monolith_matching'],
    )

    await remote_services.init()

    async with remote_services.get('redis_offers').context as redis:
        await redis.execute('flushall')

        await UniqueQueue(item_queue_name, redis).add_items(item_ids_to_match)

    admin = confluent_kafka.admin.AdminClient({
        'bootstrap.servers': config['KAFKA']['bootstrap.servers']
    })

    client_candy = ClientCandy(remote_services, None, config)

    admin.delete_topics([client_candy.topic_kafka_candidate_embedding],
                        operation_timeout=30, request_timeout=60)

    admin.delete_topics([client_candy.topic_kafka_item_matched],
                        operation_timeout=30, request_timeout=60)

    admin.delete_topics([client_candy.topic_kafka_item_not_matched],
                        operation_timeout=30, request_timeout=60)

    admin.delete_topics([client_candy.topic_kafka_item_unknown],
                        operation_timeout=30, request_timeout=60)

    admin.delete_topics([client_candy.topic_kafka_item_no_candidates_found],
                        operation_timeout=30, request_timeout=60)

    return remote_services, config


@pytest.mark.asyncio
async def test_input():
    with patch('candy.logic.implementation.client.ClientCandy._get_queue_name') as mocked_queue_name:
        mocked_queue_name.return_value = ITEM_QUEUE_KEY

        test_item_ids = ["123", 234, "345", 456, "567", 678, "789"]

        remote_services, config = await init_test(test_item_ids, ITEM_QUEUE_KEY)

        candy = ClientCandy(remote_services, None, config)
        await candy.init_queue_tasks()

        item_ids = await candy.input()

        item_ids_in_process = await candy.get_remains_from_process_queue()

        assert len(test_item_ids) == len(item_ids)

        assert len(item_ids_in_process) == len(test_item_ids)


@pytest.mark.asyncio
async def test_process_items_candidate():
    counter = MagicMock()
    counter.inc.return_value = None

    with patch("uuid.uuid4") as mocked_uuid4, \
            patch('candy.logic.implementation.client.ClientCandy._get_items_data') as mocked_items_data, \
            patch('matching_common.clients.faiss_client.IndexerApiClient.get_candidates') as mocked_candidates_faiss, \
            patch('matching_common.clients.elastic_client.ElasticServiceClient.get_candidates') as mocked_candidates_elastic, \
            patch('candy.logic.implementation.client.ClientCandy._get_candidate_data') as mocked_candidate_data, \
            patch('candy.logic.implementation.client.ClientCandy._get_match_data') as mocked_match_data, \
            patch('candy.logic.implementation.client.ClientCandy._get_match_many_data_multi') as mocked_match_many_data_multi, \
            patch('candy.logic.candy.Candy._get_categories_info') as mocked_get_categories_info, \
            patch('candy.metrics.COUNT_METRIC') as mocked_metric, \
            patch('candy.logic.candy.logging') as mocked_logging:

        mocked_uuid4.return_value = "uuid4"

        mocked_items_data.side_effect = [_get_items_data(), _get_items_full_data(0),
                                         _get_items_full_data(1), _get_items_full_data(2),
                                         _get_items_full_data(3), _get_items_full_data(4)]

        mocked_candidates_faiss.side_effect = [_get_candidates_faiss()]
        mocked_candidates_elastic.side_effect = [_get_candidates_elastic()]

        mocked_candidate_data.side_effect = [
            _get_candidate_data(0), _get_candidate_data(1), _get_candidate_data(2), _get_candidate_data(3), _get_candidate_data(4)
        ]

        mocked_match_data.side_effect = [await _get_match_data(i) for i in range(16)]

        mocked_match_many_data_multi.side_effect = [
            {"45": await _get_match_many_data_range(0, 4)},
            {"123": await _get_match_many_data_range(4, 8)},
            {"123": await _get_match_many_data_range(8, 12)},
            {"123": await _get_match_many_data_range(12, 16)},
            {"123": await _get_match_many_data_range(16, 17)}
        ]

        mocked_metric.labels.return_value = counter

        test_item_ids = ["123", 234, "345", 456, "567", 678, 789]

        mocked_candy = MagicMock()
        mocked_candy.language = os.environ.get("CANDY_LANGUAGE")
        queue_name = ClientCandy._get_queue_name(mocked_candy, "uQueue-offerMatching-ng-offers")
        remote_services, config = await init_test(test_item_ids, queue_name)

        mocked_get_categories_info.side_effect = [
            _get_categories_info(config["SUPPORTED_CATEGORIES"].split(','))
        ]

        candy = ClientCandy(remote_services, None, config)

        await candy.init_queue_tasks()
        _ = await candy.update_categories_info(list(candy.supported_categories))

        item_ids = await candy.input()

        await candy.process_items(item_ids)

        await _assert_redis_queues(remote_services, config.get('LANGUAGE'), queue_name, candy.get_process_queue_name(),
                                   client_candy=candy)

        (await remote_services.get("kafka").producer()).flush()

        await _assert_kafka_messages(remote_services, client_candy=candy)

        await remote_services.close_all()

        counter.inc.assert_has_calls([call(1)])
        mocked_logging.warning.assert_has_calls([
            call('Catalogue returned less items than should.'),
            call('Some items were not processed, can not obtain the data for these items: [\'567\']')
        ])


async def _get_items_from_redis_to_assert(remote_services: RemoteServices, type_redis: str, item_queue_name: str) \
        -> List[bytes]:
    async with remote_services.get(type_redis).context as redis:
        item_ids = await redis.execute('lrange', item_queue_name, 0, -1)

        return item_ids


async def _assert_redis_queues(remote_services: RemoteServices, language: str, queue_name: str, process_queue_name: str,
                               client_candy: ClientCandy) -> None:

    assert [b'567'] == await _get_items_from_redis_to_assert(remote_services, 'redis_offers', queue_name)

    assert [] == await _get_items_from_redis_to_assert(remote_services, 'redis_offers', process_queue_name)

    monolith_queue_name = '{}-{}'.format(client_candy.topic_monolith_redis, language)

    expected = [
        {
            "candidate_id": "19",
            "final_decision": "yes",
            "item_id": "678"
        },
        {
            "candidate_id": "23",
            "final_decision": "yes",
            "item_id": "234"
        }
    ]

    result = await _get_items_from_redis_to_assert(remote_services, 'redis_monolith_matching', monolith_queue_name)
    assert expected == [ujson.loads(r.decode()) for r in result]


async def _assert_kafka_messages(remote_services: RemoteServices, client_candy: ClientCandy) -> None:

    messages = []

    consumer = await remote_services.get('kafka').consumer()
    consumer.subscribe([
        client_candy.topic_kafka_item_matched,
        client_candy.topic_kafka_item_not_matched,
        client_candy.topic_kafka_item_unknown,
        client_candy.topic_kafka_item_no_candidates_found,
    ])

    msgs = consumer.consume(6, timeout=10)

    assert msgs is not None
    assert len(msgs) == 6

    for msg in msgs:
        messages.append(ujson.loads(msg.value()))

    messages.sort(key=lambda x: x["final_decision"] if "final_decision" in x else "0")

    assert messages == [
        {
            "id": "123",
            "match_name": "Tvoje bába",
            "shop_id": "CZE",
        },
        {
            "uuid": "uuid4",
            "final_decision": "no",
            "item": {"id": "345", "match_name": "Jeho baba", "shop_id": "CZE"},
            "final_candidate": "",
            "candidates": [
                {"id": "11", "name": "11_123", "category_id": "123", "distance": "0.011", "relevance": ""},
                {"id": "23", "name": "23_2", "category_id": "2", "distance": "0.023", "relevance": ""},
                {"id": "31", "name": "31_123", "category_id": "123", "distance": "0.031", "relevance": ""},
                {"id": "46", "name": "46_123", "category_id": "123", "distance": "0.046", "relevance": ""},
            ],
            "comparisons": [
                {
                    "id": "11",
                    "decision": "no",
                    "details": "",
                },
                {
                    "id": "23",
                    "decision": "no",
                    "details": "",
                },
                {
                    "id": "31",
                    "decision": "no",
                    "details": "",
                },
                {
                    "id": "46",
                    "decision": "no",
                    "details": "",
                },
            ],
            "possible_categories": "123,2",
            "model_info": ""
        },
        {
            "uuid": "uuid4",
            "final_decision": "unknown",
            "item": {"id": 456,  "match_name": "Nase baba", "shop_id": "CZE"},
            "final_candidate": "",
            "candidates": [
                {"id": "19", "name": "19_123", "category_id": "123", "distance": "0.019", "relevance": ""},
                {"id": "23", "name": "23_234", "category_id": "234", "distance": "0.023", "relevance": ""},
                {"id": "37", "name": "37_123", "category_id": "123", "distance": "0.037", "relevance": ""},
                {"id": "41", "name": "41_123", "category_id": "123", "distance": "0.041", "relevance": ""},
            ],
            "comparisons": [
                {
                    "id": "19",
                    "decision": "yes",
                    "details": "",
                },
                {
                    "id": "23",
                    "decision": "yes",
                    "details": "",
                },
                {
                    "id": "37",
                    "decision": "no",
                    "details": "",
                },
                {
                    "id": "41",
                    "decision": "unknown",
                    "details": "",
                },
            ],
            "possible_categories": "123,234",
            "model_info": ""
        },
        {
            "uuid": "uuid4",
            "final_decision": "unknown",
            "item": {"id": "789",  "match_name": "Jejich baba", "shop_id": "CZE"},
            "final_candidate": "",
            "candidates": [
                {"id": "42", "name": "42_123", "category_id": "123", "distance": "0.041", "relevance": ""},
            ],
            "comparisons": [
                {
                    "id": "42",
                    "decision": "unknown",
                    "details": "Decision YES, but other candidate prioritized by status or candidate is disabled",
                },
            ],
            "possible_categories": "123",
            "model_info": ""
        },
        {
            "uuid": "uuid4",
            "final_decision": "yes",
            "item": {"id": 234, "match_name": "moje bába", "shop_id": "CZE"},
            "final_candidate": "23",
            "candidates": [
                {"id": "23", "name": "23_45", "category_id": "45", "distance": "", "relevance": "0.023"},
                {"id": "24", "name": "24_342", "category_id": "342", "distance": "0.024", "relevance": ""},
                {"id": "31", "name": "31_123", "category_id": "123", "distance": "0.031", "relevance": ""},
                {"id": "48", "name": "48_123", "category_id": "123", "distance": "0.048", "relevance": ""},
            ],
            "comparisons": [
                {
                    "id": "23",
                    "decision": "yes",
                    "details": "",
                },
                {
                    "id": "24",
                    "decision": "no",
                    "details": "",
                },
                {
                    "id": "31",
                    "decision": "no",
                    "details": "",
                },
                {
                    "id": "48",
                    "decision": "no",
                    "details": "",
                },
            ],
            "possible_categories": "123,342,45",
            "model_info": ""
        },
        {
            "uuid": "uuid4",
            "final_decision": "yes",
            "item": {"id": 678, "match_name": "Vase baba", "shop_id": "CZE"},
            "final_candidate": "19",
            "candidates": [
                {"id": "19", "name": "19_123", "category_id": "123", "distance": "0.019", "relevance": ""},
                {"id": "23", "name": "23_234", "category_id": "234", "distance": "0.023", "relevance": ""},
                {"id": "38", "name": "38_123", "category_id": "123", "distance": "0.037", "relevance": ""},
                {"id": "42", "name": "42_123", "category_id": "123", "distance": "0.041", "relevance": ""},
            ],
            "comparisons": [
                {
                    "id": "19",
                    "decision": "yes",
                    "details": "Name match: namesimilarity=1.3",
                },
                {
                    "id": "23",
                    "decision": "no",
                    "details": "",
                },
                {
                    "id": "42",
                    "decision": "unknown",
                    "details": "Decision YES, but other candidate prioritized by status or candidate is disabled",
                },
                {
                    "id": "38",
                    "decision": "unknown",
                    "details": "Changed from YES due to name match of another product. ",
                },
            ],
            "possible_categories": "123,234",
            "model_info": ""
        },
    ]


async def _get_items_data() -> List[dict]:
    return [
        {
            'id': "123",
            'match_name': 'Tvoje bába',
            'shop_id': 'CZE',
        },
        {
            'id': 234,
            'match_name': 'moje bába',
            'shop_id': 'CZE',
        },
        {
            'id': "345",
            'match_name': 'Jeho baba',
            'shop_id': 'CZE',
        },
        {
            'id': 456,
            'match_name': 'Nase baba',
            'shop_id': 'CZE',
        },
        {
            'id': 678,
            'match_name': 'Vase baba',
            'shop_id': 'CZE',
        },
        {
            'id': "789",
            'match_name': 'Jejich baba',
            'shop_id': 'CZE',
        }
    ]


def _get_items_full_data(set_: int):
    data = [
        [{
            'id': 234,
            'match_name': 'moje bába',
            'shop_id': 'CZE',
        }],
        [{
            'id': "345",
            'match_name': 'Jeho baba',
            'shop_id': 'CZE',
        }],
        [{
            'id': 456,
            'match_name': 'Nase baba',
            'shop_id': 'CZE',
        }],
        [{
            'id': 678,
            'match_name': 'Vase baba',
            'shop_id': 'CZE',
        }],
        [{
            'id': "789",
            'match_name': 'Jejich baba',
            'shop_id': 'CZE',
        }]
    ][set_]

    async def async_helper():
        return data

    return async_helper()


async def _get_candidates_faiss() -> defaultdict:
    item_candidates = defaultdict(lambda: defaultdict(dict))

    item_candidates["234"] = {
        "24": Candidate(id="24", distance=0.024, source=[FAISS_CANDIDATES_SOURCE]),
        "31": Candidate(id="31", distance=0.031, source=[FAISS_CANDIDATES_SOURCE]),
        "48": Candidate(id="48", distance=0.048, source=[FAISS_CANDIDATES_SOURCE]),
    }

    item_candidates["345"] = {
        "11": Candidate(id="11", distance=0.011, source=[FAISS_CANDIDATES_SOURCE]),
        "23": Candidate(id="23", distance=0.023, source=[FAISS_CANDIDATES_SOURCE]),
        "31": Candidate(id="31", distance=0.031, source=[FAISS_CANDIDATES_SOURCE]),
        "46": Candidate(id="46", distance=0.046, source=[FAISS_CANDIDATES_SOURCE]),
    }

    item_candidates["456"] = {
        "19": Candidate(id="19", distance=0.019, source=[FAISS_CANDIDATES_SOURCE]),
        "23": Candidate(id="23", distance=0.023, source=[FAISS_CANDIDATES_SOURCE]),
        "37": Candidate(id="37", distance=0.037, source=[FAISS_CANDIDATES_SOURCE]),
        "41": Candidate(id="41", distance=0.041, source=[FAISS_CANDIDATES_SOURCE]),
    }

    item_candidates["678"] = {
        "19": Candidate(id="19", distance=0.019, source=[FAISS_CANDIDATES_SOURCE]),
        "23": Candidate(id="23", distance=0.023, source=[FAISS_CANDIDATES_SOURCE]),
        "38": Candidate(id="38", distance=0.037, source=[FAISS_CANDIDATES_SOURCE]),
        "42": Candidate(id="42", distance=0.041, source=[FAISS_CANDIDATES_SOURCE]),
    }

    item_candidates["789"] = {
        "42": Candidate(id="42", distance=0.041, source=[FAISS_CANDIDATES_SOURCE]),
    }

    return item_candidates


async def _get_candidates_elastic() -> defaultdict:

    item_candidates = defaultdict(lambda: defaultdict(dict))

    item_candidates["234"] = {
        "23": Candidate(id="23", relevance=0.023, source=[ELASTIC_CANDIDATES_SOURCE]),
    }

    item_candidates["345"] = {
        "11": Candidate(id="11", relevance=5.6, source=[ELASTIC_CANDIDATES_SOURCE]),
    }

    return item_candidates


def _get_candidate_data(_set: int) -> List[dict]:
    candidates_data = [
        [
            {
                'id': "23",
                'category_id': "45",
                'name': "23_45",
            },
            {
                'id': "24",
                'category_id': "342",
                'name': "24_342"
            },
            {
                'id': "31",
                'category_id': "123",
                'name': "31_123"
            },
            {
                'id': "48",
                'category_id': "123",
                'name': "48_123"
            },
        ],
        [
            {
                'id': "11",
                'category_id': "123",
                'name': "11_123"
            },
            {
                'id': "23",
                'category_id': "2",
                'name': "23_2",
            },
            {
                'id': "31",
                'category_id': "123",
                'name': "31_123"
            },
            {
                'id': "46",
                'category_id': "123",
                'name': "46_123"
            },
        ],
        [
            {
                'id': "19",
                'category_id': "123",
                'name': "19_123",
                'status': {"id": 11, "name": "ACTIVE"}
            },
            {
                'id': "23",
                'category_id': "234",
                'name': "23_234",
                'status': {"id": 11, "name": "ACTIVE"}
            },
            {
                'id': "37",
                'category_id': "123",
                'name': "37_123"
            },
            {
                'id': "41",
                'category_id': "123",
                'name': "41_123"
            },
        ],
        [
            {
                'id': "19",
                'category_id': "123",
                'name': "19_123",
                'status': {"id": 11, "name": "ACTIVE"}
            },
            {
                'id': "23",
                'category_id': "234",
                'name': "23_234",
            },
            {
                'id': "38",
                'category_id': "123",
                'name': "38_123",
                'status': {"id": 11, "name": "ACTIVE"}
            },
            {
                'id': "42",
                'category_id': "123",
                'name': "42_123",
                'status': {"id": 14, "name": "DISABLED"}
            },
        ],
        [
            {
                'id': "42",
                'category_id': "123",
                'name': "42_123",
                'status': {"id": 14, "name": "DISABLED"}
            },
        ],
    ]

    async def async_helper():
        return candidates_data[_set]

    return async_helper()


async def _get_match_many_data() -> List[dict]:
    return [
        {
            "match": "yes",  # 234 - 23
            "details": "",
        },
        {
            "match": "no",  # 234 - 24
            "details": "",
        },
        {
            "match": "no",  # 234 - 31
            "details": "",
        },
        {
            "match": "no",  # 234 - 48
            "details": "",
        },
        {
            "match": "no",  # 345 - 11
            "details": "",
        },
        {
            "match": "no",  # 345 - 23
            "details": "",
        },
        {
            "match": "no",  # 345 - 31
            "details": "",
        },
        {
            "match": "no",  # 345 - 46
            "details": "",
        },
        {
            "match": "yes",  # 456 - 19
            "details": "",
        },
        {
            "match": "yes",  # 456 - 23
            "details": "",
        },
        {
            "match": "no",  # 456 - 37
            "details": "",
        },
        {
            "match": "unknown",  # 456 - 41
            "details": "",
        },
        {
            "match": "yes",  # 678 - 19
            "details": "Name match: namesimilarity=1.3",
        },
        {
            "match": "no",  # 678 - 23
            "details": "",
        },
        {
            "match": "yes",  # 678 - 38
            "details": "",
        },
        {
            "match": "yes",  # 678 - 42
            "details": "",
        },
        {
            "match": "yes",  # 789 - 42
            "details": "",
        }
    ]


async def _get_match_data(_set: int) -> dict:
    res = await _get_match_many_data()
    return res[_set], None


async def _get_match_many_data_range(start, end):
    res = await _get_match_many_data()
    return res[start: end], None


async def _get_categories_info(categories):
    return [{"id": cat, "ean_required": 0, "unique_names": 0, "long_tail": 0} for cat in categories]
