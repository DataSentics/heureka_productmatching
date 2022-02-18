import time
import logging


def ready():
    with open("/tmp/readiness", "w") as f:
        f.write(str(time.time()))

    logging.debug("Ready.")


def live():
    with open("/tmp/liveness", "w") as f:
        f.write(str(time.time()))

    logging.debug("Live.")
