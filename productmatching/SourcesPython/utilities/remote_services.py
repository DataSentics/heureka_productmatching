import os

from llconfig import Config
from llconfig.converters import json as js
from occommon.security.vault import Vault
from buttstrap.remote_services import RemoteServices

from utilities.args import str2bool
from matching_common.clients.provider import ELASTIC_CANDIDATES_SOURCE

from candy.config import _prepare_kafka_connection_json_config


def init_config() -> Config:
    # no prefix, default is "_APP"
    conf = Config(env_prefix="")

    # elastic
    conf.init('VAULT_ADDR', str, "http://vault.stage.k8s.heu.cz/")
    conf.init('VAULT_TOKEN', str)
    conf.init('VAULT_ML_MOUNT_POINT', str, "matching")
    conf.init('VAULT_ELASTIC_PATH', str, "elastic")
    conf.init('ELASTIC', js)
    conf.init('ELASTIC_USERNAME', str)
    conf.init('ELASTIC_PASSWORD', str)

    if str2bool(os.environ.get("USE_GCP_ELASTIC", "false")):
        vault = Vault(address=conf['VAULT_ADDR'], token=conf['VAULT_TOKEN'])
        vault.is_authenticated()

        elastic_data = vault.get_auth_data(
            mount_point=conf['VAULT_ML_MOUNT_POINT'],
            path=conf['VAULT_ELASTIC_PATH']
        )
        conf["ELASTIC"] = {
                "hosts": [elastic_data['endpoint']]
            }
        conf['ELASTIC_USERNAME'] = elastic_data['username']
        conf['ELASTIC_PASSWORD'] = elastic_data['password']
    else:

        conf["ELASTIC"] = {
                "hosts": os.environ.get("KUBE_ELASTIC_ADDRESS", "http://matching-search-es-data.stage.k8s.heu.cz:80")
            }

    # cs2
    conf.init('CATALOGUE', str, "http://catalogue-catalogue-service2.cz.k8s.heu.cz/v1/")

    # galera
    dbpsw = os.environ.get("DEXTER_DB_PASSWORD")
    conf.init('LANGUAGE', str, "cz")
    conf.init('ENVIRONMENT', str, "production")

    conf.init('DB_HOST', str, "xtradb1-pxc.monolit")
    conf.init('DB_USER', str, "dexter_cz")
    conf.init('DB_PASSWORD', str, dbpsw)
    conf.init('DB_DB', str, "dexter_cz")
    conf.init('DB_PORT', str, 3306)

    conf.init('DEXTER_DB', js, dict(master=[
        dict(db=conf["DB_DB"],
             host=conf["DB_HOST"],
             port=int(conf["DB_PORT"]),
             user=conf["DB_USER"],
             password=conf["DB_PASSWORD"])
        ])
    )

    # kafka, used only for transfering matching_excel to kafka and to MCT
    conf.init("KAFKA_BOOTSTRAP_SERVERS", str, None)
    conf.init("KAFKA_CONSUMER_GROUP_ID", str, "productmatching-cz")

    conf.init("KAFKA_RETRIES", int, 5)
    conf.init("KAFKA_MAX_REQUESTS_PER_CONN", int, 1)
    conf.init("KAFKA_CONSUME_MAX_MESSAGES", int, 50_000)
    conf.init("KAFKA_CONSUME_TIMEOUT", int, 15)
    conf.init("KAFKA_CHECK_TOPIC_EXISTENCE", int, 1)

    return conf


async def get_remote_services(rs_list: list):
    if not any([i in rs_list for i in ['cs2', 'elastic', 'galera', 'kafka']]):
        return
    config = init_config()
    if 'kafka' in rs_list:
        _prepare_kafka_connection_json_config(config)
    params = {
        "conf": config,
    }
    if 'cs2' in rs_list:
        params["rest"] = ["catalogue"]
    if 'elastic' in rs_list:
        params["elastic"] = [ELASTIC_CANDIDATES_SOURCE]
        # for GCP elastic use:
        # params["service_kwargs"] = {ELASTIC_CANDIDATES_SOURCE: {"http_auth": (config['ELASTIC_USERNAME'], config['ELASTIC_PASSWORD'])}}
        params["service_kwargs"] = {}
        # parameters to use when using GCP elastic
        if config['ELASTIC_USERNAME'] and config['ELASTIC_PASSWORD']:
            params["service_kwargs"] = {ELASTIC_CANDIDATES_SOURCE: {"http_auth": (config['ELASTIC_USERNAME'], config['ELASTIC_PASSWORD'])}}
    if 'galera' in rs_list:
        params["mysql"] = ["dexter_db"]
    if 'kafka' in rs_list:
        params["kafka"] = ["kafka"]

    remote_services = RemoteServices(**params)

    await remote_services.init()

    return remote_services
