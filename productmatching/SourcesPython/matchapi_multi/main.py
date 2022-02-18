import os
import requests
import json
import logging

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), force=True)


if __name__ == "__main__":
    # api-endpoint
    URL = "https://gitlab.heu.cz/api/v4/projects/657/trigger/pipeline"
    uninstall = os.getenv("UNINSTALL")
    install = os.getenv("INSTALL")

    # find ids from config
    with open('data/matchapi_id_categories_mapping.json', 'r') as fr:
        matchapi_config = json.load(fr)
        matchapi_ids = " ".join([k for k in matchapi_config.keys() if k != "DISABLED"])
        if uninstall == "true" and install == "true":
            message = f"Uninstalling and installing matchapis: {matchapi_ids}"
        elif uninstall == "true" and install == "false":
            message = f"Uninstalling matchapis: {matchapi_ids}"
        elif uninstall == "false" and install == "true":
            message = f"Installing matchapis: {matchapi_ids}"
        else:
            message = "The resulting pipeline won't do anything."

        logging.info(message)

    # defining a params dict for the parameters to be sent to the API
    PARAMS = {
        'token': os.getenv("TRIGGER_TOKEN"),
        'ref': os.getenv("REF"),
        "variables[MATCHAPI_DEPLOY_JOB]": os.getenv("TARGET_ENVIRONMENT"),
        "variables[UNINSTALL]": uninstall,
        "variables[INSTALL]": install,
        "variables[IDS]": matchapi_ids,
    }
    logging.info(PARAMS)
    # sending get request and saving the response as response object
    r = requests.post(url=URL, params=PARAMS)

    # extracting data in json format
    data = r.json()

    logging.info(data)
