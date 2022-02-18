import os
import json
import logging

from collections import defaultdict
from utilities.gitlab_trigger import get_trigger_matchapi_params, get_trigger_workflow_params, trigger_pipeline, TRIGGER_URL, monitor_pipeline
from utilities.matchapi_operations import load_matchapi_config
from utilities.notify import USER_MENTIONS, send_slack_message
from utilities.s3_utils import upload_to_s3


def monitoring_consequent_actions(alerts, mlflow_client, start_retrain: bool = True):
    config_s3_address = os.getenv("MATCHAPI_CONFIG_FILE", "s3://matchapi-data-cz/matchapi_id_categories_mapping.json")

    matchapi_conf, matchapi_conf_path = load_matchapi_config(config_s3_address)

    disabled_ids = matchapi_conf.get("DISABLED")
    disabled_ids = disabled_ids.split(',') if disabled_ids else []

    models_to_disable = [c_alert['model_info'] for c_alert in alerts["critical"]]
    categories_to_disable = []

    retrain_workflow_id = 1
    retrain_trigger_values = defaultdict(list)
    for model_info in models_to_disable:
        registry_model_info = mlflow_client.get_model_version_info(model_info['name'], model_info['version'])
        if registry_model_info.current_stage == "Production":
            categories_to_disable.append(model_info['tags']['categories'])
            # gather data for retrain workflow trigger
            retrain_trigger_values['workflow_ids'].append(retrain_workflow_id)
            retrain_trigger_values["categories"].append(model_info["tags"]['categories'])
            retrain_trigger_values["preceding_cache"].append(model_info["tags"]['cache_address'])
            retrain_workflow_id += 1

    ids_to_disable = [k for k, v in matchapi_conf.items() if v in categories_to_disable and k not in disabled_ids]
    logging.warning(f"uninstalling matchapis {ids_to_disable}")

    # uninstall matchapis
    pipeline_trigger_params = get_trigger_matchapi_params(
        matchapi_ids=ids_to_disable,
        install="false",
        uninstall="true",
        target_envs=os.getenv("TARGET_ENVIRONMENT"),
    )

    matchapi_conf_new = matchapi_conf
    matchapi_conf_new["DISABLED"] = ",".join(sorted(disabled_ids + ids_to_disable))

    with open(matchapi_conf_path, 'w') as f:
        json.dump(matchapi_conf_new, f)

    logging.warning(f"uploading new matchapi conf {matchapi_conf_new}")
    upload_to_s3(config_s3_address, matchapi_conf_path)

    # get parameters to start retrain workflow
    workflow_trigger_params = get_trigger_workflow_params(
        workflow_deploy_type="retrain-workflow",
        preceding_cache_address=retrain_trigger_values["preceding_cache"],
        collector_categories=retrain_trigger_values["categories"],
        workflow_ids=retrain_trigger_values["workflow_ids"],
    )

    slack_msg = f"*Auto monitoring deactivated matchapis {','.join(ids_to_disable)} for categories {'; '.join(retrain_trigger_values['categories'])}*\n"

    if start_retrain:
        # add parameters to start retrain workflow pipeline
        pipeline_trigger_params.update(workflow_trigger_params)
        slack_msg += "Retrain pipeline triggered.\n"

        # trigger the pipeline
        pipeline_info = trigger_pipeline(pipeline_trigger_params)
        pipeline_result_msg = f"You can check the pipeline manually at {pipeline_info['web_url']}."

        # monitor the succes of the stages
        pipeline_success = monitor_pipeline(pipeline_info["id"])

        if not pipeline_success:
            pipeline_result_msg = ":boom::boom: The triggered pipeline failed or has been canceled!!!\n" + pipeline_result_msg

        slack_msg += pipeline_result_msg
    else:
        # only send message to slack, it contains command to start retraining
        args = [f"{k}={v}" for k, v in workflow_trigger_params.items()] + ["token=TOKEN"] + [f'ref={os.getenv("REF")}']
        pasted_args = " ".join([f"--form {arg}" for arg in args])
        command = f'curl --request POST {pasted_args} "{TRIGGER_URL}"'

        slack_msg += f"To retrain the models, trigger the pipeline using following curl command. *Replace the* `TOKEN` *in command* with real trigger token (listed e.g. in GitLab CICD variables).\n ```{command}``` "

    # send slack message
    send_slack_message(slack_msg, USER_MENTIONS)

    return pipeline_result_msg
