import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc

import shap
import mlflow

from xgboostmatching.models.features import numerical_features, discrete_features


class DataVisualiser:
    def __init__(self, data, labelcol: str = "label", directory: str = "", plots_prefix: str = "", log_to_mlflow: bool = True):
        self.data = data
        self.directory = directory
        self.labelcol = labelcol
        self.num_features = numerical_features
        self.discrete_features = discrete_features
        self.artifacts_to_log = []
        self.plots_prefix = ""
        self.log_to_mlflow = log_to_mlflow
        if plots_prefix:
            self.plots_prefix = f"{plots_prefix}_" if not plots_prefix.endswith("_") else f"{plots_prefix}"

    def plot_distributions(self, valuecol: str):
        if valuecol in self.data.columns:
            plt.figure()
            with sns.axes_style("darkgrid"):
                _ = sns.violinplot(x=valuecol, y=self.labelcol, data=self.data, orient="h", dropna=True).get_figure()
            output_path_fig = os.path.join(self.directory, f'{self.plots_prefix}{valuecol}_distributions.png')
            plt.savefig(output_path_fig, dpi=200)
            self.artifacts_to_log.append(output_path_fig)
        else:
            logging.info(f"'{valuecol}' is not present among columns in dataset for visualization, violin plot not created")

    def plot_ct(self, valuecol: str):
        # discrete variables are just integers converted to float
        # we have to convert them back
        if valuecol in self.data.columns:
            tab = pd.crosstab(self.data[valuecol].dropna().astype(int), self.data[self.labelcol])
            output_path_fig = os.path.join(self.directory, f'{self.plots_prefix}{valuecol}_crosstab.png')
            plt.figure()
            tab.plot.bar(rot=0)
            plt.savefig(output_path_fig, dpi=200)
            self.artifacts_to_log.append(output_path_fig)
        else:
            logging.info(f"'{valuecol}' is not present among columns in dataset for visualization, crosstab plot not created")

    def visualise_dataset(self):
        for c in self.discrete_features:
            try:
                self.plot_ct(c)
            except Exception as e:
                logging.warning(f"Exception while plotting crosstab for {c}: {e.__str__}")

        for c in self.num_features:
            try:
                self.plot_distributions(c)
            except Exception as e:
                logging.warning(f"Exception while plotting distribution for {c}: {e.__str__}")

        plt.close('all')

        if self.log_to_mlflow:
            for art in self.artifacts_to_log:
                mlflow.log_artifact(art)


class ModelVisualiser:
    def __init__(self, plot_dir):
        self.plot_dir = plot_dir

    def plot_importance(self, importances, metric, top_n=None):
        # input in the form {"feature_name": importance_metric_val}
        s = pd.Series(importances)
        s.sort_values(ascending=False, inplace=True)
        if top_n:
            s = s[:top_n]
        pdf = pd.DataFrame(s)
        pdf.reset_index(inplace=True)
        pdf.columns = ['feature_name', metric]

        with sns.axes_style("darkgrid"):
            plt.figure(num=None, figsize=(20, 18), dpi=200, facecolor='w')
            sns.barplot(y="feature_name", x=metric, data=pdf, orient="h")
            path = self.plot_dir / f"feature_importance_{metric}.png"
            plt.savefig(path)

        mlflow.log_artifact(str(path))
        plt.close('all')

    def plot_precision_recall_curves(self, pr_data, finetune_thr=False, data=None, thr_col=None, prec_label=None, rec_label=None):
        if data is None:
            precision, recall, thresholds = precision_recall_curve(pr_data[0], pr_data[1])
            auc_precision_recall = auc(recall, precision)
            prec_label = "precision"
            rec_label = "recall"
            data_name = pr_data[2]
            thresholds = np.append(thresholds, 1)
        else:
            # given 'precision-recall' like data
            precision = data[prec_label]
            recall = data[rec_label]
            thresholds = data[thr_col]
            data_name = ""

        plt.figure()
        plt.plot(thresholds, precision, label=prec_label)
        plt.plot(thresholds, recall, label=rec_label)
        plt.xlabel('threshold')
        plt.title(f"{prec_label} and {rec_label} curves - {data_name}")
        plt.legend()
        plt.axvline(x=0.5)
        path = self.plot_dir / f"{prec_label}_and_{rec_label}_{data_name}.png"
        plt.savefig(path)
        mlflow.log_artifact(str(path))

        plt.figure()
        plt.plot(recall, precision, label=f"{prec_label}-{rec_label}")
        plt.xlabel(rec_label)
        plt.ylabel(prec_label)
        plt.title(f"{prec_label}-{rec_label} curve - {data_name}")

        plt.text(x=0.05, y=min(precision)+0.02, s=f"AUCPR: {auc_precision_recall}")
        plt.legend()
        path = self.plot_dir / f"{prec_label}_x_{rec_label}_{data_name}.png"

        # find best threshold from given data and plot
        if finetune_thr:
            # calculate F1-score for various thresholds
            # possible other choice: (weighted) average based on wanted precision:coverage balance or general F-score
            fscore = (2 * precision * recall) / (precision + recall)
            # locate the index of the largest f-score
            ix = np.argmax(fscore)
            best_thr = thresholds[ix]
            print(f'Best Threshold={best_thr} with score={fscore[ix]}')
            # draw best threshold to plot
            plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best threshold')

            to_return = best_thr
        else:
            to_return = None

        plt.savefig(path)
        mlflow.log_artifact(str(path))

        plt.close('all')
        return to_return

    def plot_shap(self, model, X):
        shap_values = shap.TreeExplainer(model).shap_values(X)
        shap.summary_plot(shap_values, X)
        path = str(self.plot_dir / "shap_plot.png")
        plt.savefig(path, bbox_inches='tight')
        mlflow.log_artifact(path)
        plt.close('all')
