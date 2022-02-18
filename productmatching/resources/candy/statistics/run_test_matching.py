import json
import logging
import os
import random
import tarfile
import time
import urllib.request
from extract import main as run_extract

import boto3
import redis
from llconfig import Config


def upload_to_s3():
    logging.info("Uploading tar to s3...")
    current_time = int(time.time())
    s3_file = f'/statistics/statistics-{current_time}.tar.gz'

    endpoint_url = os.environ.get('MLFLOW_S3_ENDPOINT_URL', 'https://s3.heu.cz/')
    s3 = boto3.client("s3", endpoint_url=endpoint_url)
    s3.upload_file('statistics.tar.gz', 'mlflow', s3_file)

    tar_url = '/'.join(['s3:/', 'mlflow', s3_file])
    logging.info(f"tar uploaded to {tar_url}")
    return tar_url


def push_items_to_redis(
    queue: str,
    host: str,
    password: str
):
    logging.info("Connecting to redis...")

    if host.startswith('redis://'):
        host = host[8:]
    host = host.split(':')[0]

    client = redis.Redis(
        host=host,
        password=password
    )

    random.seed(0)

    with open('items.list', "r") as items_file:
        lines = items_file.readlines()
        random.shuffle(lines)

        for index, line in enumerate(lines):
            if not line:
                continue

            id_ = int(line)
            client.lpush(queue, id_)

            if index % 100 == 0 or index + 1 == len(lines):
                logging.info(f"Pushed {index + 1} ids.")

        logging.info('Pushed all %d ids.', len(lines))

    client.close()


def make_tar():
    logging.info("Creating tar file")
    with tarfile.open("statistics.tar.gz", "w:gz") as tar:
        tar.add('compare')
        tar.add('compare_only_final')
        tar.add('dump')


def notify_via_slack(s3_url):
    logging.info("Notifying via Slack...")
    slack_webhook_url = 'https://hooks.slack.com/services/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'
    message = {
        "text": f"Extracting stats is completed. You can download it from {s3_url}",
        # "channel": "@vaclav.kral",
    }

    req = urllib.request.Request(slack_webhook_url, json.dumps(message).encode('utf-8'))
    response = urllib.request.urlopen(req)
    response.read()


def main():
    logging.info("Started...")

    conf = Config(env_prefix='CANDY_')

    conf.init("REDIS_OFFERS_ADDRESS", str)
    conf.init("REDIS_OFFERS_PASSWORD", str)
    conf.init("REDIS_OFFERS_QUEUE", str, 'uQueue-offerMatching-ng-offers-cz')

    do_extract = conf.get("EXTRACT_FILES")

    # export current kafka output topics to set reader offset - reset queue
    if do_extract:
        run_extract()
    # push items to redis queue
    push_items_to_redis(conf.get("REDIS_OFFERS_QUEUE"), conf.get("REDIS_OFFERS_ADDRESS"), conf.get("REDIS_OFFERS_PASSWORD"))
    sleep_seconds = 60
    logging.info("Sleeping for %s", sleep_seconds)
    time.sleep(sleep_seconds)
    # extract
    if do_extract:
        run_extract()
        make_tar()
        tar_url = upload_to_s3()
        notify_via_slack(tar_url)


if __name__ == '__main__':
    logging.basicConfig(
        datefmt='%Y-%m-%d %H:%M:%S',
        format="[%(asctime)s] %(levelname)-8s - %(message)s",
        level=logging.INFO,
    )

    main()
