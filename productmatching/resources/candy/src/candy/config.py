import os
import logging

from llconfig import Config
from llconfig.converters import json, bool_like
from occommon.security.vault import Vault


def init_config() -> Config:
    conf = Config(env_prefix='CANDY_')

    conf.init('LANGUAGE', str)
    conf.init('ENVIRONMENT', str)

    conf.init('SERVICE_NAME', str, 'MLOfferMatching-Candy')

    conf.init('METRICS_PORT', int, 5000)
    conf.init('METRICS_ADDRESS', str, '0.0.0.0')

    conf.init('LOGGING', json)
    conf.init('LOGGING_CONFIG_FILE_JSON', str, os.path.join(os.path.dirname(__file__), "logging.config.json"))
    conf.init('LOGGING_JSON_FORMATTER', bool_like, True)
    conf.init('LOGGING_JSON_FORMATTER_FIELDS', str, '%(asctime) %(levelname) %(name) %(message)')

    conf.init('SENTRY_DSN', str)
    conf.init('SENTRY_ENVIRONMENT', str, conf["ENVIRONMENT"])
    conf.init('SENTRY_RELEASE', str)

    conf.init('ITEMS_RANGE_SIZE', int, 15)
    conf.init('CANDIDATES_LIMIT', int, 10)
    conf.init('MAX_WEIGHT', int, 10)
    conf.init('NO_ITEMS_SLEEP', int, 5)
    conf.init('EMBEDDING_QUEUE_CHECK_SLEEP', int, 15)
    conf.init('EMBEDDING_QUEUE_MAX_UPDATE_DELAY', int, 10)
    conf.init('EMBEDDING_READY_TIMEOUT', int, 2 * 60)

    # With all active dog food indexed, minimal found distance:
    #
    # "Přepravní box 70x52x52" cm           -> 0.52
    # "DVD přehrávač s TV tunerem"          -> 0.53
    # "SIGMA SD14"                          -> 0.28
    # "Těstoviny extrudované 10kg natural"  -> 0.173
    # "Calibra Basic Junior hovězí konzerva"-> 0.122
    # "ACANA Puppy & Junior"                -> 0.052
    # "Magnum masová směs cat 855g"         -> 0.041
    # "Magnum kočka masová směs 855g"       -> 0.041
    # "Ovesné vločky 15kg"                  -> 0.036
    # "Royal Canin Mini Adult"              -> 0.033
    #
    # Found distances are logged with candidates
    # Cat food etc. should be probably trained along dogs
    conf.init("DISTANCE_THRESHOLD", float, 0.15)

    conf.init("SUPPORTED_CATEGORIES", str)

    # whether imagesimilarity is used in matchapi
    # downloads also image_url and external_image_url and sends them to matchapi
    conf.init('USE_IMAGESIMILARITY', int, 0)
    conf.init('REMOVE_LONGTAIL', bool_like, False)

    conf.init('MAX_RETRY', int, 5)
    conf.init('MAX_MINUTES', int, 190)

    conf.init('CANDIDATES_PROVIDERS', str)

    conf.init('CATALOGUE', str)
    conf.init('FAISS', str)

    conf.init('ELASTIC_ADDRESS', str)
    conf.init('ELASTIC', json,  dict(
        hosts=[conf['ELASTIC_ADDRESS']]
    ))
    conf.init('ELASTIC_USERNAME', str, "")
    conf.init('ELASTIC_PASSWORD', str, "")
    conf.init('ELASTIC_CANDIDATES_INDEX', str)

    conf.init('REDIS_OFFERS_ADDRESS', str, 'redis://redis')
    conf.init('REDIS_OFFERS_PASSWORD', str, None)
    conf.init('REDIS_OFFERS_DB', int, 0)

    conf.init('REDIS_OFFERS', json, dict(
        address=conf['REDIS_OFFERS_ADDRESS'],
        password=conf['REDIS_OFFERS_PASSWORD'],
        db=conf['REDIS_OFFERS_DB'],
        minsize=1,
        maxsize=2
    ))

    conf.init('REDIS_MONOLITH_MATCHING_ADDRESS', str, 'redis://redis')
    conf.init('REDIS_MONOLITH_MATCHING_PASSWORD', str, None)
    conf.init('REDIS_MONOLITH_MATCHING_DB', int, 0)

    conf.init('REDIS_MONOLITH_MATCHING', json, dict(
        address=conf['REDIS_MONOLITH_MATCHING_ADDRESS'],
        password=conf['REDIS_MONOLITH_MATCHING_PASSWORD'],
        db=conf['REDIS_MONOLITH_MATCHING_DB'],
        minsize=1,
        maxsize=2
    ))

    conf.init("KAFKA_BOOTSTRAP_SERVERS", str, None)
    conf.init("KAFKA_CONSUMER_GROUP_ID", str, f"candy-{conf['LANGUAGE']}")
    conf.init("KAFKA_SASL_USERNAME", str)
    conf.init("KAFKA_SASL_PASSWORD", str)
    conf.init("KAFKA_RETRIES", str)
    conf.init("KAFKA_MAX_REQUESTS_PER_CONN", str)

    conf.init("KAFKA_CONSUME_MAX_MESSAGES", int, 50_000)
    conf.init("KAFKA_CONSUME_TIMEOUT", int, 15)
    conf.init("KAFKA_CHECK_TOPIC_EXISTENCE", int, 1)

    conf.init("CANDIDATE_REQUIRED_FIELDS", list, [
        "id", "category_id", "name", "prices",
        "slug", "category_slug",
        "attributes.id", "attributes.name", "attributes.value", "attributes.unit",
        "eans", "shops", "status"])

    conf.init("ITEM_REQUIRED_FIELDS", list, [
        "id", "match_name", "price",  "url",
        "attributes.id", "attributes.name", "attributes.value", "attributes.unit",
        "parsed_attributes.name", "parsed_attributes.value", "parsed_attributes.unit",
        "ean", "shop_id"])

    # Fields to use if using imagesimilarity
    if conf["USE_IMAGESIMILARITY"]:
        # image_url for product image
        conf["CANDIDATE_REQUIRED_FIELDS"].append("image_url")
        # image_url and external_image_url for offer images, external url for items for case img_url returns 404
        conf["ITEM_REQUIRED_FIELDS"].extend(["image_url", "external_image_url"])

    conf.init("TOPIC_REDIS", str, 'uQueue-offerMatching-ng-offers')  # `-LANGUAGE` suffix is added automatically
    conf.init("TOPIC_MONOLITH_REDIS", str, 'uQueue-offerMatching-ng-matched')  # `-LANGUAGE` suffix is added automatically
    conf.init("WRITE_RESULTS_TO_MONOLITH", bool_like, False)
    conf.init("TOPIC_KAFKA_CANDIDATE_EMBEDDING", str, 'matching-ng-candidate-embedding-cz')
    conf.init("TOPIC_KAFKA_ITEM_MATCHED", str, 'matching-ng-item-matched-cz')
    conf.init("TOPIC_KAFKA_ITEM_NOT_MATCHED", str, 'matching-ng-item-new-candidate-cz')
    conf.init("TOPIC_KAFKA_ITEM_UNKNOWN", str, 'matching-ng-unknown-match-cz')
    conf.init("TOPIC_KAFKA_ITEM_NO_CANDIDATES_FOUND", str, 'matching-ng-no-candidates-cz')

    conf.init('EXTRACT_FILES', bool_like, False)

    conf.init('MATCHAPI_BASE', str)
    conf.init('MATCHAPI_CONFIG_FILE', str)
    conf.init('MATCHAPI_DEFAULT_UNKNOWN', bool_like, False)

    conf.init('FEATURE_OC', bool_like, False)
    conf.init('PRIORITIZE_STATUS', bool_like, False)


    conf.init('VAULT_ENABLED', bool_like, False)

    conf.load()

    if conf['VAULT_ENABLED']:
        _enrich_conf_with_auth_data(conf)

    _prepare_kafka_connection_json_config(conf)

    return conf


def _prepare_kafka_connection_json_config(config: Config):
    if (servers := config["KAFKA_BOOTSTRAP_SERVERS"]) not in (None, ""):
        config.init("KAFKA", json, {
            "bootstrap.servers": servers,
            "producer": {

            },
            "consumer": {
                "group.id": config["KAFKA_CONSUMER_GROUP_ID"],
                "default.topic.config": {
                    "auto.offset.reset": "earliest"
                },
            },
        })
    else:
        raise ValueError("Kafka bootstrap servers not provided.")

    if (sasl_username := config.get("KAFKA_SASL_USERNAME")) not in (None, ""):
        config["KAFKA"]["sasl.username"] = sasl_username
        config["KAFKA"]["sasl.mechanisms"] = "PLAIN"
        config["KAFKA"]["security.protocol"] = "SASL_SSL"
    else:
        logging.warning("Kafka sasl username not set.")

    if (sasl_password := config.get("KAFKA_SASL_PASSWORD")) not in (None, ""):
        config["KAFKA"]["sasl.password"] = sasl_password
    else:
        logging.warning("Kafka sasl password not set.")

    if (kafka_retries := config.get("KAFKA_RETRIES")) not in (None, ""):
        config["KAFKA"]["retries"] = kafka_retries

    if (kafka_max_req_per_conn := config.get("KAFKA_MAX_REQUESTS_PER_CONN")) not in (None, ""):
        config["KAFKA"]["max.in.flight.requests.per.connection"] = kafka_max_req_per_conn


def _enrich_conf_with_auth_data(conf):
    conf.init('VAULT_ADDR', str)
    conf.init('VAULT_TOKEN', str)
    conf.init('VAULT_OC_MOUNT_POINT', str)
    conf.init('VAULT_ML_MOUNT_POINT', str)
    conf.init('VAULT_KAFKA_PATH', str)
    conf.init('VAULT_REDIS_PATH', str)
    conf.init('VAULT_ELASTIC_PATH', str)
    conf.init('VAULT_S3_PATH', str)

    vault = Vault(address=conf['VAULT_ADDR'], token=conf['VAULT_TOKEN'])
    vault.is_authenticated()

    if conf['VAULT_KAFKA_PATH'] and conf['VAULT_OC_MOUNT_POINT']:
        kafka_data = vault.get_auth_data(
            mount_point=conf['VAULT_OC_MOUNT_POINT'],
            path=conf['VAULT_KAFKA_PATH']
        )

        conf["KAFKA_BOOTSTRAP_SERVERS"] = kafka_data["url"]
        conf["KAFKA_SASL_USERNAME"] = kafka_data["username"]
        conf["KAFKA_SASL_PASSWORD"] = kafka_data["password"]

    if conf['VAULT_REDIS_PATH'] and conf['VAULT_OC_MOUNT_POINT']:
        redis_data = vault.get_auth_data(
            mount_point=conf['VAULT_OC_MOUNT_POINT'],
            path=conf['VAULT_REDIS_PATH']
        )

        conf["REDIS_OFFERS"]['password'] = redis_data['password']

    if conf['VAULT_ELASTIC_PATH'] and conf['VAULT_ML_MOUNT_POINT']:
        elastic_data = vault.get_auth_data(
            mount_point=conf['VAULT_ML_MOUNT_POINT'],
            path=conf['VAULT_ELASTIC_PATH']
        )

        conf["ELASTIC"] = {
            "hosts": [elastic_data['endpoint']]
        }
        conf['ELASTIC_USERNAME'] = elastic_data['username']
        conf['ELASTIC_PASSWORD'] = elastic_data['password']

    if conf['VAULT_S3_PATH'] and conf['VAULT_OC_MOUNT_POINT']:
        s3_data = vault.get_auth_data(
            mount_point=conf['VAULT_OC_MOUNT_POINT'],
            path=conf['VAULT_S3_PATH']
        )

        os.environ['AWS_ACCESS_KEY_ID'] = s3_data['access_key']
        os.environ['AWS_SECRET_ACCESS_KEY'] = s3_data['secret_key']
