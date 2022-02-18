import os
import time
import functools
import logging
import datetime
import typing as t

from slack_webhook import Slack

MLFLOW_EXPERIMENT_ID = os.environ.get("MLFLOW_EXPERIMENT_ID", "UNKNOWN").strip()
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "UNKNOWN").strip()
MLFLOW_RUN_ID = os.environ.get("MLFLOW_RUN_ID", "UNKNOWN").strip()

WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "").strip()
USER_MENTIONS = [u.strip() for u in os.environ.get("USER_MENTIONS", "").split(",") if u.strip()]


def format_mlflow_url(experiment_id: str = None, run_id: str = None):

    experiment_id = experiment_id or MLFLOW_EXPERIMENT_ID
    run_id = run_id or MLFLOW_RUN_ID

    if "UNKNOWN" in (experiment_id, MLFLOW_TRACKING_URI, run_id):
        return "unknown url"

    return f"{MLFLOW_TRACKING_URI}/#/experiments/{experiment_id}/runs/{run_id}"


def notify(name_or_func: t.Union[str, t.Callable]):
    def decorator(func):
        use_name = name_or_func or func.__name__
        logging.info(f"Will send notifications for {use_name} and mention {USER_MENTIONS}.")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            msg = f"Starting `{use_name}` ({format_mlflow_url()})\nat time `{datetime.datetime.now()}`\n with args `{args}`, kwargs `{kwargs}`.\n"
            send_slack_message(msg, USER_MENTIONS)

            value = func(*args, **kwargs)
            minutes = round((time.time() - start_time) / 60.0, 2)
            hours = round(minutes / 60, 2)

            msg = f"Ending `{use_name}` ({format_mlflow_url()})\nat time `{datetime.datetime.now()}`\n with output `{value}`\n in `{hours}` hours ({minutes} minutes).\n"
            send_slack_message(msg, USER_MENTIONS)

            return value

        return wrapper

    if callable(name_or_func):
        return decorator(name_or_func)

    else:
        return decorator


def send_slack_message(msg: str, user_mentions: list = []):
    logging.info("Notifying via Slack...")

    if not WEBHOOK_URL:
        logging.warning(f"WEBHOOK_URL not provided. {msg}")
    else:
        try:
            slack = Slack(url=WEBHOOK_URL)
            mentions = ["<" + (f"@{u}" if not u.startswith("@") else u) + ">" for u in user_mentions]

            slack.post(text=msg + " ".join(mentions))
            logging.info("Message sent.")
        except Exception:
            logging.exception("Sending slack message failed")
