import os
import asyncio
import logging

import mlflow
import json
import typing as t
import pandas as pd
from collections import Counter
from datetime import datetime, timedelta

from buttstrap.remote_services import RemoteServices
from matching_common.clients.cs2_client import CatalogueServiceClient

from utilities.args import str2bool
from utilities.galera import DbWorker
from utilities.component import compress
from utilities.helpers import split_into_batches
from utilities.model_registry.client import MlflowRegistryClient
from auto_monitoring.consequent_actions import monitoring_consequent_actions
from utilities.remote_services import get_remote_services
from auto_monitoring.alert import notify_alerts
from auto_monitoring.settings import alert_thresholds, eval_gap, eval_period, n_limit_rows

from xgboostmatching.models.features.features_conf import all_features
from xgboostmatching.models.visualize import DataVisualiser
from url_generator import UrlGenerator


class Evaluator():
    def __init__(self, remote_services: RemoteServices):
        self.min_days = 3
        self.remote_services = remote_services
        self.plots_path = os.path.join('/data', 'monitoring_plots')
        os.makedirs(self.plots_path, exist_ok=True)

    async def offer_download(self, offer_id: t.Union[str, int], status="all"):
        fields = ["id", "name", "match_name", "product_id", "url"]
        if not offer_id:
            return {}

        try:
            async with self.remote_services.get('catalogue').context as catalogue:
                items = await CatalogueServiceClient(
                    catalogue_service=catalogue
                ).get_offers([offer_id], fields, {"status": status})

            if items:
                return items[0]

        except Exception:
            logging.exception(f"CS2 gone away. Offers data not downloaded for {offer_id}.")

    async def product_download(self, product_id: t.Union[str, int], fields: list = [], status="all"):
        fields = ["id", "category_id", "name", "category.slug", "slug", "founder_offer"] if not fields else fields
        if not product_id:
            return {}

        try:
            async with self.remote_services.get('catalogue').context as catalogue:
                products = await CatalogueServiceClient(
                    catalogue_service=catalogue
                ).get_products([product_id], fields, status)

            if products:
                product = products[0]
                product["category_slug"] = product.get("category", {}).get("slug", "")
                return product
        except Exception:
            logging.exception(f"CS2 gone away. Products data not downloaded for {product_id}.")

    async def get_full_product_merge_history(self, product_id: t.Union[str, int]):
        request_product_id = product_id
        parent_ids = []
        while 1:
            try:
                async with self.remote_services.get('catalogue').context as catalogue:
                    response = await CatalogueServiceClient(
                        catalogue_service=catalogue
                    ).get_product_merge_history(request_product_id)
                if not response:
                    break
                res = response[0]
                parent_ids.append(str(res["product_id_parent"]))
                request_product_id = res["product_id_parent"]
            except Exception as e:
                logging.warning(f"get_full_product_merge_history failed for request_product_id {request_product_id} with exception {e}")
        return parent_ids

    @staticmethod
    def get_product_url(product_data: dict) -> str:
        if not product_data:
            return ""
        assert "category_slug" in product_data.keys()
        assert "slug" in product_data.keys()
        category_slug, slug = product_data["category_slug"], product_data["slug"]
        path = os.path.join(os.getcwd(), 'resources', 'candy',  'resources', 'url-generator-routes', 'routes.json')
        url = UrlGenerator(path, lang="cz", env="production")
        return url.get_url("heureka.product", category_seo=category_slug, product_seo=slug)

    @staticmethod
    def _get_model_id_from_info(model_info: dict) -> str:
        if not model_info:
            return ""
        return model_info["name"] + "_" + model_info["version"]

    async def _get_offer_n_paired_data(self, offer_id):
        offer_data = await self.offer_download(offer_id)
        if not offer_data:
            return {}, {}
        pid = offer_data.get("product_id")
        paired_product_data = await self.product_download(pid) if pid else {}
        # None may be returned
        if not paired_product_data:
            paired_product_data = {}
        return offer_data, paired_product_data

    async def _get_problem_dict(
        self, payload, candidate, offer_id, offer_data, pp_data, pp_url, model_id
    ):
        candidate_data = await self.product_download(candidate["id"])
        return {
            "decision": payload["comparisons"][str(candidate["id"])]["decision"],
            "offer_id": offer_id,
            "offer_name": payload["item"]["match_name"],
            "offer_url": offer_data["url"],
            "candidate_id": candidate["id"],
            "candidate_name": candidate["name"],
            "candidate_url": self.get_product_url(candidate_data),
            "candidate_category": candidate["category_id"],
            "paired_product_id": pp_data.get("id", ""),
            "paired_product_name": pp_data.get("name", ""),
            "paired_product_url": pp_url,
            "paired_product_category": pp_data.get("category_id", ""),
            "details": payload["comparisons"][str(candidate["id"])]["details"],
            "model_id": model_id,
        }

    def check_dates(self, rows: list, suffix: str) -> bool:
        times = [row["time"] for row in rows]
        t_max = max(times)
        t_min = min(times)
        logging.info(f"{suffix} - min date: {t_min}, max date: {t_max}")
        if (t_max - t_min).days < self.min_days:
            return False

        return True

    async def eval_offer_matched(self, offer_row: tuple) -> dict:
        """
        offer_row comes from the db as a tuple in the following format:
        (offer_id, payload, time)
        The offer_row is a result of processing of one offer by ML offermatching machinery.
        This function extracts and returns some information such as:
        - Whether the offer is currently paired.
        - ID of the model which processed the offer.
        - Whether the category was correctly estimated.
        - Whether it was matched by name or by ean or in a different way.
        - Rows for the output csv file with incorrect results.
        """
        offer_id = offer_row[0]
        payload = json.loads(offer_row[1])
        match_time = offer_row[2]
        if type(match_time) == str:
            match_time = datetime.strptime(match_time, "%Y-%m-%d %H:%M:%S")

        payload["comparisons"] = {c["id"]: {k: v for k, v in c.items() if k != "id"} for c in payload["comparisons"]}
        match_details = payload["comparisons"][str(payload["final_candidate"])]["details"]
        result = {
            "paired": 1, "model_info": payload["model_info"], "name_match": 0, "ean_match": 0,
            "correct_category": 0, "paired_in_candidates": 1, "incorrect_match_details": [], "time": match_time,
            "correct": 0, "comparisons": payload["comparisons"],
        }

        if match_details.startswith("Name match"):
            result["name_match"] = 1
        if match_details.startswith("Ean match"):
            result["ean_match"] = 1
        offer_data, paired_product_data = await self._get_offer_n_paired_data(offer_id)
        if not offer_data:
            return

        paired_product_id = offer_data.get("product_id")
        if not paired_product_id:
            result["paired"] = 0
            result["paired_in_candidates"] = 0

        # check category
        if str(paired_product_data.get("category_id")) in payload["possible_categories"].split(','):
            result["correct_category"] = 1

        final_candidate_merge_history = await self.get_full_product_merge_history(payload["final_candidate"])
        final_candidate_merge_history.append(str(payload["final_candidate"]))

        if str(offer_data["product_id"]) in final_candidate_merge_history:
            # correct match
            result["correct"] = 1
            return result
        elif not paired_product_id:
            return result  # There is no product paired to the offer right now -> we don't know any details as to why that is.
        else:
            paired_product_url = self.get_product_url(paired_product_data)
            # check whether paired product was among candidates
            candidates_ids = [c["id"] for c in payload["candidates"]]
            if offer_data["product_id"] not in candidates_ids:
                result["paired_in_candidates"] = 0

            incorrect_match_out_rows = []
            for candidate in payload["candidates"]:
                if payload["comparisons"][str(candidate["id"])]["decision"] == "yes" or str(candidate["id"]) == str(offer_data["product_id"]):
                    incorrect_match = await self._get_problem_dict(
                        payload, candidate, offer_id, offer_data, paired_product_data, paired_product_url,
                        self._get_model_id_from_info(payload["model_info"])
                    )
                    incorrect_match_out_rows.append(incorrect_match)

            result["incorrect_match_details"] = incorrect_match_out_rows
            return result

    def calc_log_metrics_matched(self, results_in: list, model_info: dict = {}):
        """
        Computation of aggregate statistics from results of `eval_offer_matched` method.
        """
        alerts = {"critical": [], "warning": []}
        suffix = "total"
        if model_info:
            suffix = self._get_model_id_from_info(model_info)
            results = [
                r for r in results_in
                if r["model_info"].get("name") == model_info["name"]
                and r["model_info"].get("version") == model_info["version"]
            ]
        else:
            # Only for complete results
            mlflow.log_metric(
                "matched_n_offers_no_modelid",
                len([r for r in results_in if not r['model_info']])
            )
            results = results_in

        n_samples = len(results)
        # METRIC: total nr. of offers matched by ML
        mlflow.log_metric(f"matched_n_offers_valid_{suffix}", n_samples)

        if not results:
            logging.info(f"No 'matched' results to evaluate. {suffix}")
            return alerts

        if not self.check_dates(results, suffix):
            logging.info(f"Less than minimal specified number of days in 'matched' results to evaluate. {suffix}")
            return alerts

        n_paired = sum([r["paired"] for r in results])
        n_unpaired = n_samples - n_paired
        # METRIC: nr. and percentage (out of total) of offers matched by ML currently unpaired
        mlflow.log_metric(f"matched_n_currently_unpaired_{suffix}", n_unpaired)
        mlflow.log_metric(f"matched_currently_unpaired_ratio_{suffix}", n_unpaired / n_samples)
        unpaired_threshold = alert_thresholds["UNPAIRED_MATCHED_RATIO_THRESHOLD"]
        if (n_unpaired / n_samples) > unpaired_threshold:
            alerts["warning"].append(
                {
                    "model_info": model_info,
                    "message": f"{suffix} - Ratio of paired matched offers is under {1 - unpaired_threshold}, value: {1 - (n_unpaired / n_samples)}"
                }
            )

        # TODO: this logic is temporary, will have to be modified when we gain access to information about
        # unpairing, merging of duplicate products and so on
        results = [r for r in results if r["paired"]]
        n_correct_matches = sum(r["correct"] for r in results)
        n_correct_category = sum([r["correct_category"] for r in results])
        # METRIC: (nr. of offers correctly paired by ML) / (nr. of offers paired by ML and currently paired)
        mlflow.log_metric(f"matched_correct_matches_ratio_{suffix}", n_correct_matches / n_paired)
        # METRIC: (nr. of offers assigned to their correct category) / (nr. of offers paired by ML and currently paired)
        mlflow.log_metric(f"matched_correct_category_ratio_{suffix}", n_correct_category / n_paired)
        matched_threshold = alert_thresholds["INCORRECT_MATCHED_RATIO_THRESHOLD"]
        if (n_correct_matches / n_samples) < matched_threshold:
            alerts["critical"].append(
                {
                    "model_info": model_info,
                    "message": f"{suffix} - Ratio of correct matches among currently paired offers is under {matched_threshold}, value: {n_correct_matches / n_samples}"
                }
            )

        incorrect_results = [r for r in results if r["incorrect_match_details"]]
        n_matched_by_name = sum([r["name_match"] for r in results])
        n_matched_by_name_correct = n_matched_by_name - sum([r["name_match"] for r in incorrect_results])
        n_matched_by_ean = sum([r["ean_match"] for r in results])
        n_matched_by_ean_correct = n_matched_by_ean - sum([r["ean_match"] for r in incorrect_results])
        # METRIC: nr. of offers paired by ML using name or ean
        mlflow.log_metric(f"matched_n_by_name_{suffix}", n_matched_by_name)
        mlflow.log_metric(f"matched_n_by_ean_{suffix}", n_matched_by_ean)
        # METRIC: (nr. of offers correctly paired by ML using name or ean) / (nr. of offers paired by ML using name or ean)
        if n_matched_by_name:
            mlflow.log_metric(f"matched_by_name_correct_ratio_{suffix}", n_matched_by_name_correct / n_matched_by_name)
        if n_matched_by_ean:
            mlflow.log_metric(f"matched_by_ean_correct_ratio_{suffix}", n_matched_by_ean_correct / n_matched_by_ean)

        # METRIC: percentage of incorrectly paired offers with the correct solution present among candidates
        # will be lower than 1 only if there are incorrect matches without currently paired product in candidates
        # applicable only to offers which are currenty paired
        n_paired_in_candidates = sum([r["paired_in_candidates"] for r in results if r["incorrect_match_details"]])
        mlflow.log_metric(f"matched_incorrect_paired_in_candidates_{suffix}", n_paired_in_candidates / len(incorrect_results))

        return alerts

    @staticmethod
    def features_from_details(details: str, features_to_find: t.List[str] = all_features):
        """
        Get feature names and values from details field, optionally choose only selected features.
        Assumes features are divided by ',' and in form 'feature_name=feature_value'. 
        """
        spl = details.split(",")
        features = {}
        for feature in spl:
            # skip non features (e.g. 'Numerical unit mismatch'), direct and ean name matches
            if "=" not in feature or "Name match" in feature or "Ean match" in feature:
                continue
            f_name, f_val = feature.split("=")
            if f_name in features_to_find:
                features[f_name] = float(f_val)
        return features

    @staticmethod
    def get_decisions_and_features(results: list, features_to_find: t.List[str] = all_features):
        dec_and_feats = []
        for offer_results in results:
            for comp in offer_results["comparisons"].values():
                parsed_features = Evaluator.features_from_details(comp["details"], features_to_find)
                if parsed_features:
                    dec_and_feats.append({
                        "decision": comp["decision"],
                        **parsed_features
                    })
        data = pd.DataFrame.from_dict(dec_and_feats)

        return data

    def get_and_visualize_features(self, result_dicts, plots_prefix: str = ""):
        # collect feature values for individual decisions
        data = self.get_decisions_and_features(result_dicts)
        if data.empty:
            logging.info(f"No decisions and features to plot for {plots_prefix}")
            return
        # draw plots, do not log here, logging at the end of the main function
        dv = DataVisualiser(
            data=data,
            labelcol="decision",
            directory=self.plots_path,
            plots_prefix=plots_prefix,
            log_to_mlflow=False,
        )
        dv.visualise_dataset()

    def model_plots_matched_offers(self, results: list):
        # TODO: currently using only currently paired offers until we do not know the reason of their unpairing, it will be explored
        paired = [r for r in results if r["paired"]]
        # get all present model versions
        model_versions = set(r["model_info"]["version"] for r in paired)
        # create plot for each model
        for mv in model_versions:
            paired_mv = [p for p in paired if p["model_info"]["version"] == mv]
            # process correctly matched offers
            correct = [p for p in paired_mv if p["correct"]]
            # collect decisions and features
            self.get_and_visualize_features(result_dicts=correct, plots_prefix=f"model_{mv}_correct_match")
            # incorrect matches
            incorrect = [p for p in paired_mv if not p["correct"]]
            self.get_and_visualize_features(result_dicts=incorrect, plots_prefix=f"model_{mv}_incorrect_match")

    def model_plots_unknown_offers(self, results: list):
        # get all present models
        model_ids = set(r["model_id"] for r in results)
        # create plot for each model
        for mid in model_ids:
            unknown_mid = [p for p in results if p["model_id"] == mid]
            # process multi-yes offers
            multi_yes = [p for p in unknown_mid if p["multi_yes"]]
            # collect decisions and features
            self.get_and_visualize_features(multi_yes, f"model_{mid}_multi_yes_match")

    async def process_matched(self, matched_rows: tuple):
        """
        Evaluation of matched offers from table `matching_ng_item_matched_cz`.
        Statistics/metrics are logged to MLFlow.
        Incorrect matches are saved in a csv file for further inspection.
        """
        results = []
        for batch_rows in split_into_batches(matched_rows, 100):
            batch_results = await asyncio.gather(*[self.eval_offer_matched(row) for row in batch_rows])
            results.extend(batch_results)

        # offer, for which no data are returned, are ommited from the evaluation
        results = [r for r in results if r is not None]
        alerts = self.calc_log_metrics_matched(results, {})
        results = [r for r in results if r.get("model_info")]

        # draw plots with features distributions per decision
        self.model_plots_matched_offers(results=results)

        # we use the full model info to be able to easily identify and disable underperforming models
        model_infos = list({self._get_model_id_from_info(r["model_info"]): r["model_info"] for r in results}.values())
        samples_counts = Counter([self._get_model_id_from_info(r["model_info"]) for r in results])
        for model_info in model_infos:
            model_alerts = self.calc_log_metrics_matched(results, model_info)
            for k, v in model_alerts.items():
                alerts[k].extend(v)

        pdf_out_rows = []
        incorrect_results = [r for r in results if r["incorrect_match_details"]][:1_000]
        if incorrect_results:
            incorrect_results = sorted(incorrect_results, key=lambda x: x["model_info"].get("version"))
            append_row = {k: "" for k in incorrect_results[0]["incorrect_match_details"][0]}
            for i_result in incorrect_results:
                pdf_out_rows.extend(i_result["incorrect_match_details"])
                pdf_out_rows.append(append_row)

        return alerts, pd.DataFrame(pdf_out_rows), samples_counts

    async def eval_offer_unknown(self, offer_row: tuple) -> dict:
        """
        offer_row comes from the db as a tuple in the following format:
        (offer_id, payload)
        The offer_row is a result of procesing of one offer by ML offermatching machinery.
        This function extracts and returns some information such as:
        - Whether the offer is currently paired.
        - ID of the model which processed the offer.
        - Whether the category was correctly estimated.
        - Whether there was match by name or by ean among multiple decisions to match.
        - Rows for the output csv file with problematic results such as multiple decisions to match.
        """
        offer_id = offer_row[0]
        payload = json.loads(offer_row[1])
        match_time = offer_row[2]

        payload["comparisons"] = {c["id"]: {k: v for k, v in c.items() if k != "id"} for c in payload["comparisons"]}
        model_id = self._get_model_id_from_info(payload["model_info"]) if payload["model_info"] else ""

        result = {
            "paired": 1, "correct_category": 0, "model_id": model_id, "paired_in_candidates": 0, "paired_decision": "",
            "multi_yes": 0, "name_match_in_multi_yes": 0, "ean_match_in_multi_yes": 0, "unsupported_category": 0, "problem_details": [], 
            "time": match_time, "comparisons": {},
        }

        offer_data, paired_product_data = await self._get_offer_n_paired_data(offer_id)
        if not offer_data:
            return

        paired_product_id = offer_data.get("product_id")
        if not paired_product_id:
            result["paired"] = 0

        yes_details = [c["details"] for c in payload["comparisons"].values() if c["decision"] == 'yes']
        problem_details = []
        if len(yes_details) > 1:
            paired_product_url = self.get_product_url(paired_product_data)
            result["multi_yes"] = 1
            result["comparisons"] = payload["comparisons"]
            for candidate in payload["candidates"]:
                if payload["comparisons"][str(candidate["id"])]["decision"] == "yes" or str(candidate["id"]) == str(paired_product_id):
                    problem_detail = await self._get_problem_dict(
                        payload, candidate, offer_id, offer_data, paired_product_data, paired_product_url, model_id
                    )
                    problem_details.append(problem_detail)

            result["problem_details"] = problem_details

        yes_details_name_match = [d for d in yes_details if d.startswith("Name match")]
        yes_details_ean_match = [d for d in yes_details if d.startswith("Ean match")]
        if len(yes_details) > 1:
            if len(yes_details_name_match) > 0:
                result["name_match_in_multi_yes"] = 1
            if len(yes_details_ean_match) > 0:
                result["ean_match_in_multi_yes"] = 1

        if payload["final_decision"] == "unsupported_category":
            result["unsupported_category"] = 1

        if paired_product_data:
            # check category
            if str(paired_product_data.get("category_id")) in payload["possible_categories"].split(','):
                result["correct_category"] = 1
            # check paired product among candidates
            if str(paired_product_id) in payload["comparisons"]:
                result["paired_in_candidates"] = 1
                result["paired_decision"] = payload["comparisons"][str(paired_product_data.get("id"))]["decision"]

        return result

    def calc_log_metrics_unknown(self, results_in: list, model_id: str = ""):
        """
        Computation of aggregate statistics from results of `eval_offer_unknown` method.
        """
        suffix = "total"
        if model_id:
            suffix = model_id
            results = [r for r in results_in if r["model_id"] == model_id]
            if not results:
                return
        else:
            results = results_in
            mlflow.log_metric("unknown_n_offers_no_modelid", len([r for r in results_in if not r['model_id']]))

        n_samples = len(results)
        # METRIC: nr. of offers with the 'unknown' result
        mlflow.log_metric(f"unknown_n_offers_valid_{suffix}", n_samples)

        if not results:
            logging.info("No 'unknown' results to evaluate.")
            return

        if not self.check_dates(results, suffix):
            logging.info(f"Less than minimal specified number of days in 'unknown' results to evaluate. {suffix}")
            return

        # TODO: n/percent of paired,
        # doesn't make sense without percentage of new products

        # TODO: n/percent of new product created
        # currently, we don't have an access to the necessary info

        # METRIC: n/percent of multiple yes
        n_multi_yes = sum(r["multi_yes"] for r in results)
        mlflow.log_metric(f"unknown_multi_yes_ratio_{suffix}", n_multi_yes / n_samples)

        # METRIC: n/percent of name_match_in_more_yes (from multiple yes)
        n_name_match_in_multi_yes = sum(r["name_match_in_multi_yes"] for r in results)
        if n_name_match_in_multi_yes:
            mlflow.log_metric(f"unknown_name_match_in_multi_yes_ratio_{suffix}", n_name_match_in_multi_yes / n_multi_yes)

        # METRIC: n/percent of ean_match_in_more_yes (from multiple yes)
        n_ean_match_in_multi_yes = sum([r["ean_match_in_multi_yes"] for r in results])
        if n_ean_match_in_multi_yes:
            mlflow.log_metric(f"unknown_ean_match_in_multi_yes_ratio_{suffix}", n_ean_match_in_multi_yes / n_multi_yes)

        # METRIC: n/percent of unsupported_category decision
        if suffix == "total":
            n_unsupported_category = sum(r["unsupported_category"] for r in results)
            mlflow.log_metric(f"unknown_unsupported_catregory_ratio_{suffix}", n_unsupported_category / n_samples)

        # the following metrics make sense only for paired offers
        results = [r for r in results if r["paired"]]
        n_paired = len(results)
        if not results:
            return

        # METRIC: n/percent of yes for paired product
        n_yes_paired = len([r for r in results if r["paired_decision"] == "yes"])
        mlflow.log_metric(f"unknown_paired_yes_ratio_{suffix}", n_yes_paired / n_paired)
        # METRIC: n/percent of unknown for paired product
        n_unknown_paired = len([r for r in results if r["paired_decision"] == "unknown"])
        mlflow.log_metric(f"unknown_paired_unknown_ratio_{suffix}", n_unknown_paired / n_paired)
        # METRIC: n/percent of no for paired product
        n_no_paired = len([r for r in results if r["paired_decision"] == "no"])
        mlflow.log_metric(f"unknown_paired_no_ratio_{suffix}", n_no_paired / n_paired)

        # METRIC: n/percent of correct_category
        n_correct_category = sum(r["correct_category"] for r in results)
        mlflow.log_metric(f"unknown_correct_category_ratio_{suffix}", n_correct_category / n_paired)

        # METRIC: n/percent of correct_category for unsupported_category
        n_uc_paired = len([r for r in results if r["unsupported_category"]])
        if n_uc_paired:
            n_uc_paired_cc = len([r for r in results if r["unsupported_category"] and r["correct_category"]])
            mlflow.log_metric(f"unknown_paired_correct_category_for_unsupported_category_ratio_{suffix}", n_uc_paired_cc / n_uc_paired)

        # METRIC: n/percent of paired_in_candidates
        n_paired_in_candidates = sum(r["paired_in_candidates"] for r in results)
        mlflow.log_metric(f"unknown_paired_in_candidates_ratio_{suffix}", n_paired_in_candidates / n_samples)

    async def process_unknown(self, decisions_unknown: tuple):
        results = []
        for batch_rows in split_into_batches(decisions_unknown, 100):
            batch_results = await asyncio.gather(*[self.eval_offer_unknown(row) for row in batch_rows])
            results.extend(batch_results)

        results = [r for r in results if r is not None]

        # draw plots with features distributions per decision 
        self.model_plots_unknown_offers(results=results)

        samples_counts = Counter([r.get("model_id", "") for r in results])
        self.calc_log_metrics_unknown(results)

        model_ids = sorted(list(set([r.get("model_id") for r in results]) - {""}))
        for model_id in model_ids:
            self.calc_log_metrics_unknown(results, model_id)

        pdf_out_rows = []
        problem_results = [r for r in results if r["problem_details"]][:1_000]
        if problem_results:
            problem_results = sorted(problem_results, key=lambda x: x["model_id"])
            # get the schema from first row of first result
            append_row = {k: "" for k in problem_results[0]["problem_details"][0]}
            for i_result in problem_results:
                pdf_out_rows.extend(i_result["problem_details"])
                pdf_out_rows.append(append_row)

        return pd.DataFrame(pdf_out_rows), samples_counts

    def calc_log_metrics_new_candidate(self, results_in: list, model_info: str = ""):
        alerts = {"warning": []}
        suffix = "total"
        if model_info:
            suffix = self._get_model_id_from_info(model_info)
            results = [
                r for r in results_in
                if r["model_info"].get("name") == model_info["name"]
                and r["model_info"].get("version") == model_info["version"]
            ]
        else:
            # Only for complete results
            mlflow.log_metric(
                "new_product_n_offers_no_modelid",
                len([r for r in results_in if not r['model_info']])
            )
            results = results_in

        n_samples = len(results)
        # METRIC: total nr. of 'new product' decisions
        mlflow.log_metric(f"new_product_n_offers_valid_{suffix}", n_samples)

        if not results:
            logging.info("No 'new_product' results to evaluate.")
            return alerts

        if not self.check_dates(results, suffix):
            logging.info(f"Less than minimal specified number of days in 'new_product' results to evaluate. {suffix}")
            return alerts

        resolved_offers = [r for r in results if r["correct_decision"] is not None]
        n_correct_decisions = sum(r['correct_decision'] for r in resolved_offers)
        correct_ratio = n_correct_decisions / len(resolved_offers)
        # METRIC: ratio of correct 'new_product' decisions
        mlflow.log_metric(f"new_product_ratio_correct_resolved_{suffix}", correct_ratio)

        incorrect_decisions_offers = [r for r in resolved_offers if r["correct_decision"] == 0]
        n_incorrect_decisions_w_right_candidate = sum(r["paired_in_candidates"] for r in incorrect_decisions_offers)
        # METRIC: ratio of incorrect decisions where the correct product was missing among the candidates
        mlflow.log_metric(f"new_product_ratio_incorrect_wo_right_candidate_{suffix}",
                          1 - (n_incorrect_decisions_w_right_candidate / len(incorrect_decisions_offers)))

        new_product_correct_threshold = alert_thresholds["NEW_PRODUCT_CORRECT_RATIO_THRESHOLD"]
        if correct_ratio < new_product_correct_threshold:
            alerts["warning"].append(
                {
                    "model_info": model_info,
                    "message": f"{suffix} - Ratio of correct new_product is under {new_product_correct_threshold}, value: {round(correct_ratio, 2)}"
                }
            )

        return alerts

    async def eval_offer_new_candidate(self, offer_row: tuple):
        offer_id = offer_row[0]
        payload = json.loads(offer_row[1])
        payload["comparisons"] = {c["id"]: {k: v for k, v in c.items() if k != "id"} for c in payload["comparisons"]}
        match_time = offer_row[2]
        if type(match_time) == str:
            match_time = datetime.strptime(match_time, "%Y-%m-%d %H:%M:%S")

        result = {
            "correct_decision": None, "paired_in_candidates": 0, "model_info": payload["model_info"],
            "incorrect_new_candidate_details": [], "time": match_time,
        }

        offer_data, paired_product_data = await self._get_offer_n_paired_data(offer_id)
        if not offer_data:
            return

        paired_product_id = offer_data.get("product_id")
        if paired_product_id:
            if str(paired_product_id) in payload["comparisons"]:
                result["paired_in_candidates"] = 1

            if paired_product_data["founder_offer"]:
                relation_time = datetime.strptime(paired_product_data["founder_offer"]["relation_time"], "%Y-%m-%dT%H:%M:%S")
                if match_time < relation_time:
                    # Offers' currently matched product did not yet exist -> the 'new_product' decision was correct.
                    result["correct_decision"] = 1
                else:
                    # Offers' currenly matched product existed at the time of ML's decision, hence the 'new_product' decision was incorrect.
                    # Just an approximation since the 'new product' might not get to the candidates index before the arrival of the new offer,
                    #     which happens after the creation of the 'new product'.
                    result["correct_decision"] = 0
                    incorrect_rows = []
                    paired_product_url = self.get_product_url(paired_product_data)
                    for candidate in payload["candidates"]:
                        incorrect_match = await self._get_problem_dict(
                            payload, candidate, offer_id, offer_data, paired_product_data, paired_product_url,
                            self._get_model_id_from_info(payload["model_info"])
                        )
                        incorrect_rows.append(incorrect_match)

                    result["incorrect_new_candidate_details"] = incorrect_rows

        return result

    async def process_new_candidate(self, decisions_new_candidate: tuple):
        # TODO: once we have info about creation of new products something interesting might be calculated
        results = []
        for batch_rows in split_into_batches(decisions_new_candidate, 100):
            batch_results = await asyncio.gather(*[self.eval_offer_new_candidate(row) for row in batch_rows])
            results.extend(batch_results)
        results = [r for r in results if r is not None]

        alerts = self.calc_log_metrics_new_candidate(results, {})
        results = [r for r in results if r.get("model_info")]

        model_infos = list({self._get_model_id_from_info(r["model_info"]): r["model_info"] for r in results}.values())
        samples_counts = Counter([self._get_model_id_from_info(r["model_info"]) for r in results])
        for model_info in model_infos:
            model_alerts = self.calc_log_metrics_new_candidate(results, model_info)
            for k, v in model_alerts.items():
                alerts[k].extend(v)

        pdf_out_rows = []
        incorrect_results = [r for r in results if r["incorrect_new_candidate_details"]][:1_000]
        if incorrect_results:
            incorrect_results = sorted(incorrect_results, key=lambda x: x["model_info"].get("version"))
            append_row = {k: "" for k in incorrect_results[0]["incorrect_new_candidate_details"][0]}
            for i_result in incorrect_results:
                pdf_out_rows.extend(i_result["incorrect_new_candidate_details"])
                pdf_out_rows.append(append_row)

        return alerts, pd.DataFrame(pdf_out_rows), samples_counts


@notify_alerts
async def main():
    remote_services = await get_remote_services(['cs2', 'galera'])

    eval_worker = Evaluator(remote_services)
    mlflow_client = MlflowRegistryClient()
    worker = DbWorker(remote_services)

    DATE_FROM = datetime.today() - timedelta(eval_gap + eval_period)
    DATE_TO = datetime.today() - timedelta(eval_gap)
    mlflow.log_param("DATE_FROM", str(DATE_FROM))
    mlflow.log_param("DATE_TO", str(DATE_TO))
    logging.info(f"{DATE_FROM}, {DATE_TO}")

    table_name = "matching_ng_item_matched"
    logging.info(f"Reading table {table_name}.")
    model_decisions_matched = await worker.read_messages(table_name, DATE_FROM, DATE_TO, limit=n_limit_rows['matched'])
    logging.info(f"Loaded table {table_name}.")
    alerts, pdf_matched, counts_matched = await eval_worker.process_matched(model_decisions_matched)

    if alerts["critical"] and str2bool(os.getenv("MATCHAPI_ALLOW_DISABLE", "false")):
        pipeline_result_msg = monitoring_consequent_actions(
            alerts=alerts,
            mlflow_client=mlflow_client,
            start_retrain=str2bool(os.getenv("TRIGGER_RETRAINING", "false"))
        )
        alerts["critical"].append(pipeline_result_msg)
    else:
        logging.info("Retraining turned off.")

    table_name = "matching_ng_unknown_match"
    logging.info(f"Reading table {table_name}.")
    model_decisions_unknown = await worker.read_messages(table_name, DATE_FROM, DATE_TO, limit=n_limit_rows['unknown'])
    logging.info(f"Loaded table {table_name}.")
    pdf_unknown, counts_unknown = await eval_worker.process_unknown(model_decisions_unknown)

    table_name = "matching_ng_item_new_candidate"
    logging.info(f"Reading table {table_name}.")
    model_decisions_new_product = await worker.read_messages(table_name, DATE_FROM, DATE_TO, limit=n_limit_rows['new_candidate'])
    logging.info(f"Loaded table {table_name}.")

    alerts_nc, pdf_new_candidate, counts_new_candidate = await eval_worker.process_new_candidate(model_decisions_new_product)
    alerts["warning"].extend(alerts_nc["warning"])

    sheets = {}
    if not pdf_matched.empty:
        sheets["matched"] = pdf_matched
    if not pdf_unknown.empty:
        sheets["unknown"] = pdf_unknown
    if not pdf_new_candidate.empty:
        sheets["new_candidate"] = pdf_new_candidate
    if sheets:
        path_problematic = os.path.join(os.getcwd(), 'data', 'problematic_results.xlsx')
        with pd.ExcelWriter(path_problematic) as writer:
            for sheet_name, pdf_sheet in sheets.items():
                pdf_sheet.to_excel(writer, sheet_name, index=False)

        mlflow.log_artifact(path_problematic)

    coverage_alerts, _ = calculate_coverage(counts_matched, counts_unknown, counts_new_candidate)
    alerts["warning"].extend(coverage_alerts)

    alerts_messages = []
    for severity in ["critical", "warning"]:
        for alert in alerts[severity]:
            alert_message = f"{severity.capitalize()} alert: {alert['message']}"
            alerts_messages.append(alert_message)
            logging.warning(alert_message)

    if not alerts_messages:
        alerts_messages = ["None"]

    # tar the plots and log the tar since there can be a lot of plots
    tar_file = os.path.join("/data", "monitoring_plots.tar.gz")
    compress(tar_file, eval_worker.plots_path)
    mlflow.log_artifact(tar_file)

    await remote_services.close_all()
    return alerts_messages


def calculate_coverage(counts_matched: dict, counts_unknown: dict, counts_new_candidate: dict):
    thresh = alert_thresholds["COVERAGE_THRESHOLD"]
    message = f"- Coverage has fallen below {thresh}, value:"
    model_ids = (set(counts_matched.keys()) | set(counts_unknown.keys()) | set(counts_new_candidate.keys())) - {""}
    alerts = []
    coverages = {}
    for model_id in model_ids:
        cm = counts_matched.get(model_id, 0)
        cn = counts_new_candidate.get(model_id, 0)
        coverage = 0
        if cm + cn > 0:
            coverage = round((cm + cn) / (cm + cn + counts_unknown.get(model_id, 0)), 2)
        coverages[model_id] = coverage
        mlflow.log_metric(f"coverage_{model_id}", coverage)
        if coverage < thresh:
            alerts.append({"message": f"{model_id} {message} {coverage}"})

    cmt = sum([v for k, v in counts_matched.items() if k != ""])
    cut = sum([v for k, v in counts_unknown.items() if k != ""])
    cnt = sum([v for k, v in counts_new_candidate.items() if k != ""])
    coverage = 0
    if cmt + cnt > 0:
        coverage = round((cmt + cnt) / (cmt + cnt + cut), 2)
    coverages["total"] = coverage
    mlflow.log_metric("coverage_total", coverage)
    if coverage < thresh:
        alerts.append({"message": f"total {message} {coverage}"})

    return alerts, coverages


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("AUTOMATIC_MONITORING")
    with mlflow.start_run():
        asyncio.run(main())
