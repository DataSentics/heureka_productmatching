import asyncio
import logging
import signal

import prometheus_client
import sentry_sdk
from buttstrap.log import init_logging
from buttstrap.remote_services import RemoteServices
from llconfig import Config

from candy import metrics
from candy.config import init_config
from candy.logic.implementation.client import ClientCandy
from candy.logic.implementation.oc_client import OcClientCandy
from candy.logic.matchapi import MatchApiManager
from matching_common.clients.provider import FAISS_CANDIDATES_SOURCE, ELASTIC_CANDIDATES_SOURCE


async def exit_signal_handler(sig: signal.Signals) -> None:
    logging.info(f"Received exit signal {sig.name}...")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    [task.cancel() for task in tasks]

    logging.info("Canceling outstanding tasks")
    await asyncio.gather(*tasks)
    logging.info("Outstanding tasks canceled")


def init_sentry(config: Config) -> None:
    """Initializes sentry sdk with all of the default integrations and Sanic as an extra"""
    logging.debug("initializing Sentry")

    cfg = config.get_namespace("SENTRY_")

    if not cfg["dsn"]:
        logging.warning("Sentry init skipped because of missing config")
        return

    sentry_sdk.init(**cfg)
    logging.info("Sentry inits OK")


async def main() -> None:
    # Ensure graceful shutdown https://www.roguelynn.com/words/asyncio-we-did-it-wrong-pt-2/
    loop = asyncio.get_event_loop()
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        loop.add_signal_handler(
            sig=s, callback=lambda s=s: asyncio.create_task(exit_signal_handler(s))
        )

    # Configure the necessary
    config = init_config()
    init_logging(config)
    init_sentry(config)

    prometheus_client.start_http_server(
        port=config["METRICS_PORT"], addr=config["METRICS_ADDRESS"], registry=metrics._REGISTRY
    )

    redis = ["redis_offers", "redis_monolith_matching"]

    if config['FEATURE_OC']:
        redis = ["redis_offers"]

    service_kwargs = {}

    if config['ELASTIC_USERNAME'] and config['ELASTIC_PASSWORD']:
        service_kwargs.update({
            ELASTIC_CANDIDATES_SOURCE: {
                "http_auth": (config['ELASTIC_USERNAME'], config['ELASTIC_PASSWORD'])
            }
        })

    rest = ["catalogue", FAISS_CANDIDATES_SOURCE]

    remote_services = RemoteServices(
        config,
        rest=rest,
        kafka=["kafka"],
        redis=redis,
        elastic=[ELASTIC_CANDIDATES_SOURCE],
        service_kwargs=service_kwargs
    )
    await remote_services.init()

    matchapi_manager = MatchApiManager(config)
    # we want to wait for the first discovery to complete before proceeding
    _ = await asyncio.wait_for(matchapi_manager.discover(), timeout=None)

    if config['FEATURE_OC']:
        candy = OcClientCandy(remote_services, matchapi_manager, config)
    else:
        candy = ClientCandy(remote_services, matchapi_manager, config)

    # starts the infinite loop of periodical re-discovering of available MatchAPIs
    # the eventual changes to the discovered services will propagate deeper to candy
    task = asyncio.create_task(matchapi_manager.discover_loop())
    _ = asyncio.as_completed(task)

    await candy.init_queue_tasks()
    await candy.work()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except asyncio.CancelledError:
        logging.info("Main task cancelled")
    except Exception:
        logging.exception("Something unexpected happened")
    finally:
        logging.info("Shutdown complete")
