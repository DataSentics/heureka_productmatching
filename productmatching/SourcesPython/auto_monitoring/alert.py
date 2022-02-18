import os
import functools
import logging
import typing as t

import opsgenie_sdk
from opsgenie_sdk import ApiException

OPSGENIE_API_KEY = os.environ.get("OPSGENIE_API_KEY", "").strip()
MENTIONS = "\n @ml-purple"
ENVIRONMENT = os.getenv("TARGET_ENVIRONMENT")


def opsgenie_alert(alert_message: str, priority: str = "P1"):
    if not alert_message:
        return
    conf = opsgenie_sdk.Configuration()
    conf.api_key["Authorization"] = OPSGENIE_API_KEY

    alert_api = opsgenie_sdk.AlertApi(opsgenie_sdk.ApiClient(conf))

    description = alert_message + MENTIONS
    if ENVIRONMENT:
        description = f"Environment: {ENVIRONMENT}\n" + description

    try:
        alert_payload = opsgenie_sdk.CreateAlertPayload(
            message="ML automatic monitoring",
            description=description,
            responders=[
                {"name": "ML Purple", "type": "team"}
            ],
            priority=priority,
        )
        _ = alert_api.create_alert(alert_payload)
    except ApiException as err:
        logging.exception(f"Exception when calling AlertApi->create_alert: {err}")


def notify_alerts(name_or_func: t.Union[str, t.Callable]):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            alerts_messages = await func(*args, **kwargs)
            critical_alerts = [am for am in alerts_messages if "Critical" in am]
            warning_alerts = [am for am in alerts_messages if "Warning" in am]
            opsgenie_alert("\n".join(critical_alerts))
            opsgenie_alert("\n".join(warning_alerts), "P3")

        return wrapper

    if callable(name_or_func):
        return decorator(name_or_func)

    else:
        return decorator
