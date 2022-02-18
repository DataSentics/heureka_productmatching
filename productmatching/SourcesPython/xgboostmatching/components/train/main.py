import os
import re
import argparse
import logging
import mlflow
import json
import numpy as np
import pandas as pd
import typing as t
import xgboost as xgb

from pathlib import Path
from utilities.component import process_inputs
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from scipy import stats

from utilities.notify import notify
from utilities.args import str_or_none
from utilities.logger_to_file import log_to_file_and_terminal
from xgboostmatching.components.dataset.utils import XGB_INFO_COLS
from xgboostmatching.components.train.utils import parse_parameters, MLFlowCallback
from xgboostmatching.models import features
from xgboostmatching.models.visualize import DataVisualiser, ModelVisualiser
from xgboostmatching.utils import load_data_from_paths


def filter_data_by_index(X: pd.DataFrame, y: pd.DataFrame, indices: np.array):
    ind_intersect = list(set(indices).intersection(set(X.index.to_list())))

    X_flt = X.loc[ind_intersect, :]
    y_flt = y.loc[ind_intersect].to_numpy()

    return X_flt, y_flt


def augment_test_dataset(data: pd.DataFrame, labelcol: str = "label"):
    """
    With --train-size = 0.8, there should be at least 6 samples out of which at least 2 positive and 2 negative.
    During testing, this might cause problems so we try to circumvent this issue by augmenting the new dataset.
    The resulting dataset will make the xgb model to be useless.
    """
    min_samples = 6
    min_label_count = 2
    warn = False
    n_pos_samples = sum(data[labelcol])
    n_to_add_labels = {
        1: max(0, min_label_count - n_pos_samples),
        0: max(0, min_label_count - len(data) + n_pos_samples),
    }
    for label, n_to_add in n_to_add_labels.items():
        if n_to_add > 0:
            row = data.head(1).copy()
            row["label"] = [label]
            data = pd.concat([data] + [row] * int(n_to_add))
            warn = True

    if len(data) < min_samples:
        rows = data.head(min_samples - len(data))
        data = pd.concat([data, rows])
        warn = True

    if warn:
        logging.warning(
            "The input dataset was augmented to have at least 6 samples, of which at least 2 for each label (1/0). The resulting model is useless!!!"
        )

    return data


def _add_updated_features(data: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    data_extra = load_data_from_paths(args.input_datasets_extra)
    if data_extra.empty:
        logging.info("No extra datasets present")
        return data
    # in order to prevent some possible problems during testing
    data_extra = augment_test_dataset(data_extra).sort_values('label', ascending=False)

    cols_to_drop = [c for c in data.columns if c in data_extra.columns and c not in XGB_INFO_COLS]
    logging.info(f"Using updated values for following features: {cols_to_drop}")

    return data.drop(columns=cols_to_drop).merge(data_extra, how='inner', on=XGB_INFO_COLS)


@notify
def xgboostmatching_train(args: argparse.Namespace):

    parameters = parse_parameters(args)
    eval_metric = parameters["eval_metric"][-1]

    model_dir = Path(args.data_directory / "xgboostmatching_model")
    model_dir.mkdir(exist_ok=True)

    # load data
    new_data = load_data_from_paths(args.input_datasets)
    # augment dataset for testing
    new_data = augment_test_dataset(new_data).sort_values('label', ascending=False)
    n_pos_new_data = sum(new_data.label > 0)
    # process preceding data, returns empty dataframe if there are no input datasets
    preceding_data = load_data_from_paths(args.preceding_input_datasets)
    n_pos_preceding_data = 0
    if not preceding_data.empty:
        preceding_data.sort_values('label', ascending=False, inplace=True)
        preceding_data = augment_test_dataset(preceding_data).sort_values('label', ascending=False)
        n_pos_preceding_data = sum(preceding_data.label > 0)

    mlflow.log_metric("n_samples_new", len(new_data))
    mlflow.log_metric("n_samples_preceding", len(preceding_data))

    data = _add_updated_features(pd.concat([new_data, preceding_data], ignore_index=True), args)

    # array to use during stratification
    # it is constant if no preceding data or it has two labels based on counts of preceding and new data
    stratify_ar = np.array(
        ["created_1"] * n_pos_new_data +
        ["created_0"] * (len(new_data) - n_pos_new_data) +
        ["preceding_1"] * n_pos_preceding_data +
        ["preceding_0"] * (len(preceding_data) - n_pos_preceding_data)
    )
    stratify_df = pd.DataFrame(stratify_ar)

    mlflow.log_metric("n_samples", len(data))
    mlflow.log_metric("n_positive_samples", sum(data.label))
    mlflow.log_metric("n_negative_samples", len(data) - sum(data.label))

    dataset_profile_dir = os.path.join(args.data_directory, "xgb_dataset_plots")
    os.makedirs(dataset_profile_dir, exist_ok=True)
    DV = DataVisualiser(data, "label", dataset_profile_dir)
    DV.visualise_dataset()

    # drop features present in data and not specified in featrues_conf.py
    drop_cols = [col for col in data.columns if col not in features.all_features and col != "label"]
    logging.info(f"Dropping following columns not specified in `features_conf.py`: {drop_cols}")

    # drop columns where the most frequent value occupies more than 99% values
    # applied to discrete features only
    drop_allowed = features.discrete_features
    max_counts = data[drop_allowed].apply(pd.Series.value_counts).apply(max)
    drop_cols_99 = [col for col, v in max_counts.items() if v / len(data) >= 0.99 and col != "label"]

    # TODO: add also some correlation criterion
    logging.info(f"Dropping following columns with 99% of one value: {drop_cols_99}")
    drop_cols += drop_cols_99
    data.drop(columns=drop_cols, inplace=True)

    X = pd.DataFrame(data.drop(columns='label'))
    y = data['label']
    feature_names = list(X.columns)
    logging.info(f"Used features: {feature_names}")

    del data

    if args.randomized_search_iter != -1:
        # do randomized grid search
        y = y.to_numpy()
        # define classifier
        spw = (len(y) - sum(y)) / sum(y)
        clf_xgb = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric=eval_metric, booster="gbtree", scale_pos_weight=spw
        )
        # define parameters in search
        param_dist = {
            'n_estimators': stats.randint(300, 1000),
            'learning_rate': stats.uniform(0.01, 0.6),
            'subsample': stats.uniform(0.3, 0.7),
            'max_depth': [3, 4, 5, 6, 7, 8, 9],
            "max_delta_step": list(range(0, 20)),
            'min_child_weight': [1, 2, 3, 4],
            "reg_lambda": stats.uniform(0.0001, 1000),
            "reg_alpha": stats.uniform(0.0001, 0.8)
        }
        logging.info(f"Randomized Search CV with '{param_dist}'")

        # metrics to monitor
        metrics = ['roc_auc', 'accuracy', 'f1']
        # stratified cross validation
        kfold = StratifiedKFold(n_splits=5, shuffle=True)

        clf = RandomizedSearchCV(
            clf_xgb,
            param_distributions=param_dist,
            cv=kfold,
            n_iter=args.randomized_search_iter,
            scoring=metrics,
            error_score=0,
            verbose=10,
            n_jobs=1,  # default value, but probably better to be explicit
            refit='roc_auc'
        )
        clf.fit(X, y)

        # log results
        cv_results = pd.DataFrame(clf.cv_results_)
        cv_results.to_csv(model_dir / "cv_results.csv", index=False)
        mlflow.log_artifact(str(model_dir / "cv_results.csv"))

        # tag metrics of chosen model
        for metric in metrics:
            value = cv_results[f"mean_test_{metric}"][clf.best_index_]
            mlflow.set_tag(f"best_mean_test_{metric}", value)

        # best model, as type Booster, the same as if fit directly
        model = clf.best_estimator_.get_booster()
        model.set_attr(feature_names='|'.join(feature_names))
        model.save_model(model_dir / "best.xgb")
        best_model = model
        logging.info(f"Best params: {clf.best_params_}")

        # save best parameters
        with open(model_dir / "best_params.json", 'w') as fp:
            json.dump(clf.best_params_, fp)
        mlflow.log_artifact(str(model_dir / "best_params.json"))

        y_pr = [(y, model.predict(xgb.DMatrix(data=X)), "all")]
    else:
        mlflow.set_tag("xgboost_parameters", args.parameters)

        # fit without search
        # split, stratify by data type to ensure similar portion of new/preceding data
        # augment the test size for very small dataframes
        test_size = max(
            1 - args.train_size,
            len(set(stratify_df.loc[:, 0])) / len(X)
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=stratify_df)

        # check start of string to capture both the original and augmented (possibly in tests) data
        created_ind = stratify_df.index[np.char.startswith(stratify_ar, "created")].to_list()
        X_train_new, y_train_new = filter_data_by_index(X_train, y_train, created_ind)
        X_test_new, y_test_new = filter_data_by_index(X_test, y_test, created_ind)

        preceding_ind = stratify_df.index[np.char.startswith(stratify_ar, "preceding")].to_list()
        X_train_preceding, y_train_preceding = filter_data_by_index(X_train, y_train, preceding_ind)
        X_test_preceding, y_test_preceding = filter_data_by_index(X_test, y_test, preceding_ind)

        logging.info(f"New data: train: {len(X_train_new)}, test: {len(X_test_new)}, total: {len(X_train_new)+len(X_test_new)}, originally: {sum(stratify_ar=='created')}")
        logging.info(f"Preceding data: train: {len(X_train_preceding)}, test: {len(X_test_preceding)}, total: {len(X_train_preceding)+len(X_test_preceding)}, originally: {sum(stratify_ar=='preceding')}")

        # train with new and preceding data together,
        # other setups (e.g. putting larger weight for new data) led to similar or worse results

        # load data into DMatrix, needed for xgb.fit (to be able to log the best model continuously)
        dtrain = xgb.DMatrix(data=X_train, label=y_train)
        dtest = xgb.DMatrix(data=X_test, label=y_test)

        eval_list = [
            (dtrain, "train"),
            (dtest, "eval"),  # Last one is used for early stopping
        ]

        # train, it returns model from the last iteration
        spw = (len(y_train) - sum(y_train)) / sum(y_train)
        parameters["scale_pos_weight"] = spw
        model = xgb.train(
            parameters,
            dtrain,
            args.iterations,
            evals=eval_list,
            callbacks=[
                MLFlowCallback("eval-aucpr", model_dir, dtest, feature_names),
                xgb.callback.early_stop(100, maximize=True, verbose=True),
            ],
        )
        # save last model as best if it wasn't saved in MLFlowCallback
        if "best.xgb" not in os.listdir(model_dir):
            logging.info("No best model saved, saving the last as best")
            model.set_attr(feature_names='|'.join(feature_names))
            model.save_model(model_dir / "best.xgb")
            best_model = model
        else:
            best_model = xgb.Booster()
            best_model.load_model(str(model_dir / "best.xgb"))

        y_pr = [
            (y_train, best_model.predict(xgb.DMatrix(data=X_train)), "train_all"),
            (y_test, best_model.predict(xgb.DMatrix(data=X_test)), "test_all"),
            (y_train_preceding, best_model.predict(xgb.DMatrix(data=X_train_preceding)), "train_preceding"),
            (y_test_preceding, best_model.predict(xgb.DMatrix(data=X_test_preceding)), "test_preceding"),
            (y_train_new, best_model.predict(xgb.DMatrix(data=X_train_new)), "train_new"),
            (y_test_new, best_model.predict(xgb.DMatrix(data=X_test_new)), "test_new")
        ]

    # log best model
    mlflow.log_artifact(str(model_dir / "best.xgb"))

    MV = ModelVisualiser(model_dir)

    # log and plot feature importances
    base_features = [f for f in features.all_features if f not in drop_cols]
    for fi in ['weight', 'gain', 'total_gain']:
        dic = best_model.get_score(importance_type=fi)
        if dic:
            path = str(model_dir / f"{fi}.json")
            with open(path, "w") as f:
                json.dump(dic, f)
            mlflow.log_artifact(path)
            # plot top 50 features
            MV.plot_importance(dic, fi, 50)
            # basic features importance
            dic_base = {k: v for k, v in dic.items() if k in base_features}
            if dic_base:
                MV.plot_importance(dic_base, fi)

    # draw precison-recall curves
    for pr_data in y_pr:
        if len(pr_data[0]):
            logging.info(f"Visualising {pr_data[2]}")
            MV.plot_precision_recall_curves(pr_data)

    # draw shap figures
    # the try except block is just a safeguard
    try:
        MV.plot_shap(best_model, X)
    except ImportError as e:
        logging.warning(f"Shap library related import error, no shap figures generated: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), force=True)
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-datasets", required=True)
    parser.add_argument("--input-datasets-extra", type=str_or_none, required=True)
    parser.add_argument("--parameters", required=True)  # Format: name=value,name=value
    parser.add_argument("--randomized-search-iter", type=int, default=-1)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--data-directory", type=Path, default="/data")
    parser.add_argument("--train-size", type=float, default=0.8)
    parser.add_argument("--n-components", type=str, default='mle')
    parser.add_argument("--preceding-input-datasets", type=str_or_none, default=None)
    parser.add_argument("--preceding-data-directory", default="/preceding_data")

    args = parser.parse_args()

    assert (args.randomized_search_iter == -1) | (args.randomized_search_iter > 0), "Set randomized_search_iter -1 or number of iterations to do"
    args.input_datasets = process_inputs(args.input_datasets.split("@"), args.data_directory)
    args.input_datasets_extra = process_inputs(args.input_datasets_extra.split("@"), args.data_directory) if args.input_datasets_extra else []

    # get preceding dataset uris or empty list if no specified
    args.preceding_input_datasets = args.preceding_input_datasets.split("@") if args.preceding_input_datasets else []
    # prepare data, different data directory required since it would replace the files from args.input_datasets
    args.preceding_input_datasets = process_inputs(args.preceding_input_datasets, args.preceding_data_directory)

    if re.match(r'^\d+$', args.n_components):
        args.n_components = int(args.n_components)

    logging.info(args)

    with mlflow.start_run():
        log_to_file_and_terminal(xgboostmatching_train, args)
