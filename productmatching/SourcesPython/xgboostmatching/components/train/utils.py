import argparse
import logging
import mlflow
import json
import numpy as np
import pandas as pd
import pickle as pk
import xgboost as xgb
import typing as t

from pathlib import Path
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.metrics import accuracy_score


def parse_parameters(args: argparse.Namespace) -> dict:
    parsed: t.Dict[str, t.Union[str, t.List[str]]] = {}

    for parameter in args.parameters.split(","):
        name, value = [x.strip() for x in parameter.split("=")]

        if name in parsed and isinstance(parsed[name], str):
            parsed[name] = [parsed[name], value]

        elif name in parsed and isinstance(parsed[name], list):
            parsed[name] = [*parsed[name], value]

        else:
            parsed[name] = value

    em = parsed.get("eval_metric", None)
    em = [em] if em is not None else []

    # Last one is used for early stopping, auc if the metric isn't specified in params
    parsed["eval_metric"] = ["error", "logloss", "auc"] + em
    logging.info(f"parsed xgb parameters: {parsed}")
    return parsed


class MLFlowCallback:
    def __init__(
        self,
        watched: str,
        data: Path,
        dtest: xgb.DMatrix,
        features: list,
        maximize: bool = True,
        threshold: float = 0.5,
    ):
        self.best = float("-inf") if maximize else float("inf")
        self.watched = watched
        self.data = data
        self.features = features
        self.maximize = maximize
        self.dtest = dtest
        self.threshold = threshold

    def __call__(self, env):
        mlflow.log_metric("iteration", env.iteration, step=env.iteration)

        for name, value in env.evaluation_result_list:
            mlflow.log_metric(name, value, step=env.iteration)

            if name == self.watched and (
                (self.maximize and value > self.best)
                or (not self.maximize and value < self.best)
            ):
                self.best = value
                print(f"New best {name}: {self.best}.")

                env.model.set_attr(feature_names='|'.join(self.features))
                env.model.save_model(self.data / "best.xgb")
                mlflow.log_artifact(str(self.data / "best.xgb"))

                mlflow.set_tag(f"best_eval_{name}", self.best)

                test_pred = [
                    1 if p >= self.threshold else 0
                    for p in env.model.predict(self.dtest)
                ]
                mlflow.set_tag(
                    "best_eval_accuracy",
                    accuracy_score(self.dtest.get_label(), test_pred)
                )
