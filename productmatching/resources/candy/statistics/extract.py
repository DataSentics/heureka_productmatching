import csv
import logging
import ujson
import confluent_kafka
import typing
from contextlib import contextmanager
from dataclasses import dataclass
from llconfig import Config
from pathlib import Path
from url_generator import UrlGenerator


RETRY_MAX_COUNT = 12


@dataclass
class ExportFiles:
    dump: typing.TextIO
    writer_candidates: csv.DictWriter = None
    writer_final: csv.DictWriter = None


MATCHING_OUTPUT_TOPICS = [
    "matching-ng-item-matched-cz",
    "matching-ng-item-new-candidate-cz",
    "matching-ng-unknown-match-cz",
]

# This topic has messages in different format {id: x, match_name: y}
NO_CANDIDATES_FOUND_TOPIC = "matching-ng-no-candidates-cz"

TSV_COLUMNS = [
    'content_decision',
    'decision',
    'item_name',
    'candidate_name',
    'item_url',
    'candidate_url',
    'item_id',
    'candidate_id',
    'category_id',
    'uuid',
    'details',
    'candidate_source'
]


def get_product_url(category_slug: str, product_slug: str) -> str:

    path_file = Path(__file__).resolve()
    url = UrlGenerator(path_file.parents[1] / 'resources' / 'url-generator-routes' / 'routes.json',
                       lang="cz", env="production")
    return url.get_url("heureka.product", category_seo=category_slug, product_seo=product_slug)


def create_item_part_of_row(data: dict) -> dict:
    """
    Returns dict which contains general item's info.
    """
    row = {
        'uuid': data["uuid"],
        'item_id': data['item']['id'],
        'item_name': data['item']['data']['match_name'],
        'item_url': data['item']['data']['url'],
    }

    return row


def create_candidate_part_of_row(row: dict, comparisons_candidate: dict, final_decision=None) -> dict:
    """
    Returns dict for placing data of candidate from comparisons list into a tsv_candidates file row with certain tsv_candidates columns.
    """
    if comparisons_candidate['candidate']['data']:
        candidate_url = get_product_url(comparisons_candidate['candidate']['data']['category_slug'],
                                        comparisons_candidate['candidate']['data']['slug'])

        row['decision'] = comparisons_candidate['final_decision'] if final_decision else comparisons_candidate['decision']
        row['details'] = comparisons_candidate['comparisons'][0]['details'] if final_decision else comparisons_candidate['details']
        row['candidate_name'] = comparisons_candidate['candidate']['data']['name']
        row['candidate_id'] = comparisons_candidate['candidate']['data']['id']
        row['category_id'] = comparisons_candidate['candidate']['data']['category_id']
        row['candidate_url'] = candidate_url
        row['candidate_source'] = comparisons_candidate['candidate']['source']

    return row


def get_consumer():
    conf = Config(env_prefix='CANDY_')

    conf.init("KAFKA_BOOTSTRAP_SERVERS", str, "kafka:29092")  # 29092 inside docker
    conf.init("KAFKA_SASL_USERNAME", str)
    conf.init("KAFKA_SASL_PASSWORD", str)

    kafka_config = {}

    sasl_username = conf.get("KAFKA_SASL_USERNAME")
    if sasl_username is not None:
        kafka_config["sasl.username"] = sasl_username
        kafka_config["sasl.mechanisms"] = "PLAIN"
        kafka_config["security.protocol"] = "SASL_SSL"

    sasl_password = conf.get("KAFKA_SASL_PASSWORD")
    if sasl_password is not None:
        kafka_config["sasl.password"] = sasl_password

    consumer = confluent_kafka.Consumer({
        "bootstrap.servers": conf.get('KAFKA_BOOTSTRAP_SERVERS'),
        "group.id": f"topic_extractor",
        "default.topic.config": {
            "enable.auto.commit": "true",
            "auto.offset.reset": "earliest",
        },
        **kafka_config
    })

    return consumer


def read_topics(topics: [str]):
    consumer = get_consumer()

    logging.info(f"Subscribing to {topics}.")
    consumer.subscribe(topics)

    consumed = 0
    retry = 0
    while True:
        if not consumed % 20 and consumed:
            logging.info(f"Consumed {consumed}")

        msg = consumer.poll(5.0)

        if msg is None:
            retry += 1
            if retry > RETRY_MAX_COUNT:
                logging.info(f"Reached maximum retires {RETRY_MAX_COUNT}, consumed {consumed}, breaking.")
                break
            else:
                logging.debug("Waiting...")
                continue

        if msg.error():
            logging.error(f"Consumer error: {msg.error()}, consumed {consumed}, breaking.")
            break

        consumed += 1
        retry = 0
        yield msg

    consumer.unsubscribe()


@contextmanager
def get_topic_files():
    writers = {}
    files = []

    for topic in MATCHING_OUTPUT_TOPICS + [NO_CANDIDATES_FOUND_TOPIC]:
        files.append(open(f"dump/{topic}.txt", "w"))
        dump_file = files[-1]

        writer_candidates, writer_final = None, None
        if topic != NO_CANDIDATES_FOUND_TOPIC:
            files.append(open(f"compare/{topic}.tsv", 'w'))
            writer_candidates = csv.DictWriter(files[-1], fieldnames=TSV_COLUMNS, delimiter='\t', restval='')
            writer_candidates.writeheader()

            files.append(open(f"compare_only_final/{topic}.tsv", 'w'))
            writer_final = csv.DictWriter(files[-1], fieldnames=TSV_COLUMNS, delimiter='\t', restval='')
            writer_final.writeheader()

        writers[topic] = ExportFiles(dump_file, writer_candidates, writer_final)

    yield writers

    for f in files:
        f.close()


def retrieve_compare_topics_data():
    with get_topic_files() as files:
        for msg in read_topics(MATCHING_OUTPUT_TOPICS + [NO_CANDIDATES_FOUND_TOPIC]):
            topic = msg.topic()
            value = msg.value().decode('utf-8')
            files[topic].dump.write(value + "\n")

            if topic == NO_CANDIDATES_FOUND_TOPIC:
                continue

            data = ujson.loads(value)
            row = create_item_part_of_row(data)

            if data["candidate"] is not None:
                final_decision_row = create_candidate_part_of_row(row, data, final_decision=True)
                files[topic].writer_final.writerow(final_decision_row)

            for candidate in data["comparisons"]:
                comparisons_data = create_candidate_part_of_row(row, candidate)
                files[topic].writer_candidates.writerow(comparisons_data)

            files[topic].writer_candidates.writerow({})

        logging.info('The script has finished parsing')


def main():
    logging.info("Started...")

    retrieve_compare_topics_data()


if __name__ == '__main__':
    logging.basicConfig(
        datefmt='%Y-%m-%d %H:%M:%S',
        format="[%(asctime)s] %(levelname)-8s - %(message)s",
        level=logging.INFO,
    )

    main()
