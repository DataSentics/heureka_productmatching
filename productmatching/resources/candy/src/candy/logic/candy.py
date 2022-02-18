"""Core business logic for Candy, needs a communication-layer implementation above it."""
import asyncio
import logging
import tenacity
import itertools
import dataclasses
import time
import uuid
import re

import numpy as np

from abc import abstractmethod

from typing import List, Any, Union, Iterable
from collections import defaultdict
from enum import Enum

from buttstrap.remote_services import RemoteServices
from llconfig import Config

from matching_common.clients.cs2_client import CatalogueServiceClient
from matching_common.clients.ocs_client import OneCatalogueServiceClient
from matching_common.clients.base_client import RETRY_ATTEMPTS

from candy.logic.matchapi import MatchApiClient, MatchApiManager
from candy.logic.providers.candidates_provider import CandidatesProvider, AVAILABLE_CANDIDATES_SERVICE
from candy.logic.providers.candidates_monitor import CandidatesMonitor
from candy import metrics


class Decision(Enum):
    yes = "yes"
    no = "no"
    unknown = "unknown"


class Candy:

    def __init__(
            self,
            remote_services: RemoteServices,
            matchapi_manager: MatchApiManager,
            config: Config
    ) -> None:
        """
        Args:
            remote_services:         For communication purposes
            config:                  Config
        """
        self.remote_services = remote_services
        self.matchapi_manager = matchapi_manager
        self.language = config['LANGUAGE']
        self.elastic_candidates_index = config['ELASTIC_CANDIDATES_INDEX']
        self.candidates_limit = config['CANDIDATES_LIMIT']
        self.max_weight = config['MAX_WEIGHT']
        self.distance_threshold = config['DISTANCE_THRESHOLD']
        self.no_items_sleep = config["NO_ITEMS_SLEEP"]
        self.kafka_consume_max_messages = config["KAFKA_CONSUME_MAX_MESSAGES"]
        self.kafka_consume_timeout = config["KAFKA_CONSUME_TIMEOUT"]
        self.candidate_required_fields = config["CANDIDATE_REQUIRED_FIELDS"]
        self.item_required_fields = config["ITEM_REQUIRED_FIELDS"]
        self.check_topic_existence = config["KAFKA_CHECK_TOPIC_EXISTENCE"]
        self.max_retry = config["MAX_RETRY"]
        self.max_minutes = config["MAX_MINUTES"]

        self.topic_redis = config['TOPIC_REDIS']
        self.topic_monolith_redis = config['TOPIC_MONOLITH_REDIS']
        self.write_result_to_monolith = config['WRITE_RESULTS_TO_MONOLITH']
        self.topic_kafka_candidate_embedding = config['TOPIC_KAFKA_CANDIDATE_EMBEDDING']
        self.topic_kafka_item_matched = config['TOPIC_KAFKA_ITEM_MATCHED']
        self.topic_kafka_item_not_matched = config['TOPIC_KAFKA_ITEM_NOT_MATCHED']
        self.topic_kafka_item_unknown = config['TOPIC_KAFKA_ITEM_UNKNOWN']
        self.topic_kafka_item_no_candidates_found = config['TOPIC_KAFKA_ITEM_NO_CANDIDATES_FOUND']

        self.candidates_monitor = CandidatesMonitor(config['CANDIDATES_PROVIDERS'].replace(' ', '').split(','))
        self.candidates_providers = self._get_candidates_providers(config['CANDIDATES_PROVIDERS'])

        self.catalogue_client = OneCatalogueServiceClient if config['FEATURE_OC'] else CatalogueServiceClient

        self.feature_oc = config['FEATURE_OC']
        self.prioritize_status = config['PRIORITIZE_STATUS']

        self.remove_longtail = config["REMOVE_LONGTAIL"]

        if self.feature_oc:
            self.supported_categories = set()
        else:
            self.supported_categories = set(config['SUPPORTED_CATEGORIES'].split(','))

        self.categories_info = {}

    @abstractmethod
    async def ready(self) -> bool:
        pass

    @abstractmethod
    async def fetch_embedding(self):
        pass

    @abstractmethod
    async def input(self) -> None:
        pass

    @abstractmethod
    async def output_discarded(self, payload: dict) -> None:
        pass

    @abstractmethod
    async def output_item_matched(self, payload: dict) -> None:
        pass

    @abstractmethod
    async def output_item_not_matched(self, payload: dict) -> None:
        pass

    @abstractmethod
    async def output_item_unknown(self, payload: dict) -> None:
        pass

    @abstractmethod
    async def output_item_no_candidates_found(self, payload: dict) -> None:
        pass

    @abstractmethod
    async def rollback(self, payload: Iterable):
        """Rollback processed data."""

    @abstractmethod
    async def rollback_item(self, item: Union[bytes, str, int]):
        """Rollback item from process queue into matching queue."""

    @abstractmethod
    async def ack(self, payload: Any):
        """Ack data as processed."""

    @abstractmethod
    async def get_remains_from_process_queue(self):
        """Returns items from process queue. You can use the method to find out items that were not processed."""

    @staticmethod
    def _prioritize_status(yes, no, unknown, feature_oc: bool = False):
        """
        Changes model decisions based on candidate status if some candiates paired, giving priority to candidates with statuses
        based on defined hierarchy. Also unpairs paired disabled candidates.
        """
        # hierarchy of statuses, format 'status_id': rank. Lower ranks have priority,
        # candidates with lowest rank status are paired, decisions of candidates with higher rank are changed to unknown

        if feature_oc:
            statuses_hierarchy = {
                "true": 0,  # active
                "false": 1,  # inactive
            }
            # status to be unpaired, inactive/disabled/...
            # put None to use only statuses hierarchy
            inactive_status = "false"

            yes_statuses = [y["candidate"]["data"].get("status", None) for y in yes]
        else:
            statuses_hierarchy = {
                11: 0,  # ACTIVE
                12: 0,  # ACTIVE_HIDDEN_CATEGORY
                15: 1,  # NOT_FOR_SALE
                16: 1,  # NOT_FOR_SALE_HIDDEN_CATEGORY
                13: 2,  # NOT_APPROVED
                14: 3,  # DISABLED
            }
            # status to be unpaired, inactive/disabled/...
            # put None to use only statuses hierarchy
            inactive_status = 14

            yes_statuses = [y["candidate"]["data"].get("status", {}).get("id", None) for y in yes]

        # continue only if we have status of all candidates
        if None in yes_statuses:
            return yes, no, unknown
        statuses_lvl = np.array(list(map(statuses_hierarchy.get, yes_statuses)))
        unique_statuses_lvl = sorted(list(set(statuses_lvl)))

        idx_to_rewrite = np.array([])
        # prioritize only if statuses with multiple priorities present
        if len(unique_statuses_lvl) > 1:
            highest_prio_status = np.min(unique_statuses_lvl)
            # indices to rewrite
            idx_to_rewrite = np.where(statuses_lvl > highest_prio_status)[0]
        # only disabled candidates, change to unknown decision
        elif len(unique_statuses_lvl) == 1 and inactive_status and unique_statuses_lvl[0] == statuses_hierarchy[inactive_status]:
            idx_to_rewrite = np.where(statuses_lvl == unique_statuses_lvl[0])[0]
        # rewrite decisions
        msg = "Decision YES, but other candidate prioritized by status or candidate is disabled"
        for yes_idx_to_change in idx_to_rewrite:
            yes_to_change = yes[yes_idx_to_change]
            yes_to_change["decision"] = "unknown"
            yes_to_change["details"] = msg + f", {yes_to_change['details']}" if yes_to_change['details'] != "" else msg

        new_unknown = [y for y in yes if y["decision"] == Decision.unknown.value] + unknown
        new_yes = [y for y in yes if y["decision"] == Decision.yes.value]

        return new_yes, no, new_unknown

    @staticmethod
    def _prioritize_name_match(yes_decisions):
        name_match_indexes = []
        for i, decision in enumerate(yes_decisions):
            if 'Name match' in decision["details"] and 'under threshold' not in decision["details"]:
                name_match_indexes.append(i)

        # eiter 0 or >=2 name matches, do not prioritize
        if len(name_match_indexes) != 1:
            return yes_decisions, []

        # found exactly one name match
        # change other name matches to unknown
        # add this information to details
        additional_unknown = []
        name_match_index = name_match_indexes[0]
        for i, decision in enumerate(yes_decisions):
            if i == name_match_index:
                yes_new = [decision]
            else:
                additional_unknown.append({
                    "decision": Decision.unknown.value,
                    "candidate": decision.get("candidate"),
                    "details": "Changed from YES due to name match of another product. " + decision["details"],
                    **{k: v for k, v in decision.items() if k not in ["decision", "candidate", "details"]}
                })
        return yes_new, additional_unknown

    def _get_candidates_providers(self, candidates_providers_to_use: str) -> List[str]:
        candidates_providers = []

        for provider in candidates_providers_to_use.replace(', ', ',').split(','):
            if provider in AVAILABLE_CANDIDATES_SERVICE and provider not in candidates_providers:
                candidates_providers.append(provider)

        if not candidates_providers:
            raise ValueError("Candidate providers have to not be empty. Define at least one.")

        return candidates_providers

    async def _get_candidates(self, items: List[dict], item_ids: List[Union[int, str]]) -> defaultdict:
        candidates_providers_params = {
            'distance_threshold': self.distance_threshold,
            'index_name': self.elastic_candidates_index,
            'candidates_metric': metrics.CANDIDATES_METRIC,
            'app_name': metrics.APP_NAME,
        }

        if not self.candidates_providers:
            return defaultdict(lambda: defaultdict(dict))

        return await CandidatesProvider(
            self.candidates_providers,
            self.remote_services,
            self.language,
            self.feature_oc
        ).get_candidates(items, item_ids, self.candidates_limit, **candidates_providers_params)

    @staticmethod
    def _get_final_message(
        final_decision: str, item: dict, final_candidate: dict,
        candidates: list, comparisons: list, pc_message: str, model_info: dict
    ):
        def sim_to_str(sim):
            return str(sim) if sim else ""
        return {
            "uuid": str(uuid.uuid4()),
            "final_decision": final_decision,
            "item": {"id": item["id"], "match_name": item["match_name"], "shop_id": item["shop_id"]},
            "final_candidate": final_candidate.get("id", "") if final_candidate is not None else "",
            "candidates": [
                {
                    "id": c["data"]["id"], "name": c["data"]["name"], "category_id": c["data"]["category_id"],
                    "distance": sim_to_str(c["distance"]), "relevance": sim_to_str(c["relevance"])
                }
                for c in candidates if c.get("data")
            ],
            "comparisons": [
                {"id": c["candidate"]["id"], "decision": c["decision"], "details": c["details"]}
                for c in comparisons
            ] if comparisons else "",
            "possible_categories": pc_message,
            "model_info": model_info if model_info else "",
        }

    def _get_category_estimate(self, candidates: List[dict]) -> dict:
        def trans(x):
            return self.distance_threshold - x
        category_estimates = defaultdict(dict)
        # using (self.distance_threshold - x) transformation, this will assign high values to most similar products and vice versa
        # any monotonically decreasing function on [0, inf) would do, but we assume, that the distribution of candidates' distances is somehow uniform
        # just for consistency of ordering for both relevance and distance
        candidates_distance = [c for c in candidates if c.get("distance") and c.get("data")]
        total_trans_distance = sum([trans(c.get("distance")) for c in candidates_distance])
        categories_distance = {c["data"]["category_id"] for c in candidates_distance}
        for category in categories_distance:
            prob_category = sum(
                [trans(c.get("distance")) for c in candidates_distance if c["data"]["category_id"] == category]
            ) / total_trans_distance
            category_estimates["distance"][str(category)] = prob_category

        # relevance - the higher the better
        candidates_relevance = [c for c in candidates if c.get("relevance") and c.get("data")]
        total_relevance = sum([c.get("relevance") for c in candidates_relevance])
        categories_relevance = {c["data"]["category_id"] for c in candidates_relevance}
        for category in categories_relevance:
            prob_category = sum([c.get("relevance") for c in candidates_relevance if c["data"]["category_id"] == category]) / total_relevance
            category_estimates["relevance"][str(category)] = prob_category

        return category_estimates

    def _process_category_estimates(self, candidates: list):
        # We want to somehow determine the most probable catogry/ies of incoming offer.
        # There has to be some limit set on the distance of the candidates from FAISS,
        # otherwise we might get category estimates from totaly non-similar candidates.
        category_estimates = self._get_category_estimate(candidates)
        # take only categories with high probability
        # TODO: test the behavior having a high number of indexed categories, probably nothing special will happen

        possible_categories = set()
        for dist_metric in ["distance", "relevance"]:
            cat_probs = category_estimates.get(dist_metric, {})
            for cat_id, prob in cat_probs.items():
                possible_categories.add((cat_id, prob))

        possible_categories_message = ','.join(sorted([p[0] for p in possible_categories]))

        return possible_categories, possible_categories_message

    def _log_item_missing_fileds(self, item_full):
        n_item_missing_fields = 0
        for k, v in item_full.items():
            if v is None:
                n_item_missing_fields += 1
                metrics.OFFER_MISSING_FIELDS_COUNT_METRIC.labels(self.language, str(k)).observe(1)
                logging.info(f"{str(item_full['id'])} missing value for {k}")
        if n_item_missing_fields > 0:
            metrics.OFFER_MISSING_FIELDS_COUNT_METRIC.labels(self.language, 'any').observe(n_item_missing_fields)

    async def _process_item_candidates(self, item, candidates):
        if not candidates:
            logging.debug(f"No candidates found for item id {item['id']}, sending to monolith.")
            # TODO: currently we does not allow write to monolith redis
            # await self.output_discarded(item["id"])
            await self.ack(self._item_redis_id(item))
            await self.output_item_no_candidates_found(item)
            metrics.COUNT_METRIC.labels('no_candidates_received', self.language, "none").inc()
            return

        # candidate retrieval monitoring
        await self.candidates_monitor.monitor_incoming_candidates(list(candidates.values()), self.language)

        try:
            candidates_datas = await self._get_candidate_data(
                list(candidates.keys()),
                fields=self.candidate_required_fields,
            )
        except Exception:
            logging.exception(f"Exception while _get_candidate_data for candidates {candidates.keys()}.")
            metrics.COUNT_METRIC.labels('exception', self.language, "none").inc()
            metrics.COUNT_EXCEPTION_METRIC.labels('_get_candidate_data', self.language).inc()
            return

        if not candidates_datas:
            logging.warning(f"Something wrong, no candidates data was returned. IDs {candidates.keys()}")
            await self.ack(self._item_redis_id(item))
            metrics.COUNT_METRIC.labels('no_candidates_data', self.language, "none").inc()
            return

        for data in candidates_datas:
            assert str(data["id"]) in candidates
            candidates[str(data["id"])].data = data

        candidates_out = list(candidates.values())
        candidates_out.sort(key=lambda c: c.relevance if c.relevance else c.distance if c.distance else 0)
        candidates_out = [
            {
                **{k: v for k, v in dataclasses.asdict(candidate).items() if k != "vector"},
                # Distance as string, otherwise ujson wont be able to serialize it
                "distance": candidate.distance if candidate.distance else "",
                "relevance": candidate.relevance if candidate.relevance else "",
            }
            for candidate in candidates_out
        ]

        return candidates_out

    async def _process_item(self, item, item_to_candidates):

        id_str = str(item["id"])
        st = time.time()
        logging.debug(f'starting {id_str}')
        candidates = item_to_candidates[id_str]

        candidates = await self._process_item_candidates(item, candidates)
        if candidates is None:
            return

        if self.feature_oc:
            possible_categories_message = ''
        else:
            possible_categories, possible_categories_message = self._process_category_estimates(candidates)
            logging.debug(f"Item {item['id']} possible categories: {possible_categories}")
            if not possible_categories:
                message = self._get_final_message(
                    "unknown", item, {}, candidates, [], possible_categories_message, {}
                )
                await self.output_item_unknown(message)
                await self.ack(self._item_redis_id(item))
                return
            # TODO: Currently taking only the one highest ranked category,
            # since we don't know what to do with multiple matching results
            top_category_estimate = sorted(possible_categories, key=lambda x: x[1])[-1][0]
            allowed_possible_categories = [top_category_estimate]

        candidates_count = len(candidates)
        # filter removed candidates
        candidates = [candidate for candidate in candidates if candidate["data"]]
        candidates = await self.category_info_to_candidates_data(candidates)

        # optionally filter out candidates from longtail categories and do not pair offers with predicted longtail category
        if self.remove_longtail:
            # find out if predicted category is longtail or not
            category_info = await self.get_category_info(int(allowed_possible_categories[0]))

            # do not pair longtail category, only ack item
            if category_info["long_tail"]:
                message = self._get_final_message(
                    "unsupported_category", item, None, candidates, [], possible_categories_message, {}
                )
                log_message = f"Offer {item['id']} rejected beacuse of longtail category: {possible_categories_message}."
                logging.info(log_message)
                metrics.COUNT_METRIC.labels('offer_category_not_accepted', self.language, "none").inc()
                await self.ack(item["id"])
                return

            # non longtail category, only remove longtail candidates
            candidates_count_before_longtail = len(candidates)
            candidates = [candidate for candidate in candidates if not candidate["long_tail"]]
            if len(candidates) != candidates_count_before_longtail:
                logging.info(f"Removed {candidates_count_before_longtail - len(candidates)} candidates from longtail categories")

        if removed_candidates_count := candidates_count - len(candidates):
            metrics.COUNT_METRIC.labels('removed_candidates', self.language, "none").inc(removed_candidates_count)

        if not candidates:
            logging.debug(f"No existing candidate found for item id {id_str}, sending to monolith.")
            # TODO: currently we do not allow write to monolith redis
            # await self.output_discarded(item["id"])
            await self.output_item_no_candidates_found(item)
            await self.ack(self._item_redis_id(item))
            metrics.COUNT_METRIC.labels('no_candidates_received', self.language, "none").inc()
            return

        try:
            if self.feature_oc:
                item_full = [item]
            else:
                item_full = await self._get_items_data(
                    [item["id"]],
                    fields=self.item_required_fields,
                )
        except Exception:
            metrics.COUNT_METRIC.labels('exception', self.language, "none").inc()
            metrics.COUNT_EXCEPTION_METRIC.labels('_get_items_data', self.language).inc()
            return

        logging.debug("For item '%s' found %d candidates", item['match_name'], len(candidates))

        if not item_full:
            logging.warning("Something went wrong, no item data was returned.")
            return

        assert len(item_full) == 1
        item_full = item_full[0]
        if not self.feature_oc:
            item_full["categories"] = allowed_possible_categories
        # log number of mising fields from "self.item_required_fields"
        self._log_item_missing_fileds(item_full)

        logging.debug(
            f"Asking for match of item ({id_str})."
        )

        logging.debug([c['id'] for c in candidates])
        try:
            final_decision, final_candidate, comparisons, model_info = await self.get_matches(item_full, candidates)
            logging.info(f'Item {id_str} decision "{final_decision.value}".')
        except Exception:
            logging.exception(f"Exception while get_matches for item id {id_str}.")
            metrics.COUNT_METRIC.labels('exception', self.language, "none").inc()
            metrics.COUNT_EXCEPTION_METRIC.labels('get_matches', self.language).inc()
            return

        message = self._get_final_message(
            str(final_decision.value), item, final_candidate, candidates, comparisons, possible_categories_message, model_info
        )

        if final_decision == Decision.yes:
            await self.output_item_matched(message)

        elif final_decision == Decision.no:
            await self.output_item_not_matched(message)

        elif final_decision == Decision.unknown:
            await self.output_item_unknown(message)

        else:
            raise ValueError(f"Invalid final decision {final_decision}.")

        await self.ack(self._item_redis_id(item))
        logging.info(f'ending {id_str} in {time.time() - st}')

    async def process_items(self, item_ids: List[str]) -> None:
        logging.debug("Starts to find candidates for item matching...")
        try:
            items = await self._get_items_data(item_ids, ["id", "match_name", "shop_id"])
        except Exception:
            await self.rollback(item_ids)

            logging.exception(f"Exception while _get_items_data for item ids {item_ids}.")
            metrics.COUNT_METRIC.labels('exception', self.language, "none").inc()
            metrics.COUNT_METRIC.labels('rollback_items_to_queue', self.language, "none").inc(len(item_ids))
            metrics.COUNT_EXCEPTION_METRIC.labels('_get_items_data', self.language).inc()
            return

        if not items:
            logging.warning(f"Catalogue returned no items for {item_ids}.")
            for item_id in item_ids:
                await self.ack(item_id)
            return

        elif len(items) != len(item_ids):
            logging.warning("Catalogue returned less items than should.")
            if self.feature_oc:
                item_ids_from_catalogue = [f"product:{item['shop_id']}:{item['id']}" for item in items]
            else:
                item_ids_from_catalogue = [item['id'] for item in items]

            for id_ in [item_id for item_id in item_ids if item_id not in item_ids_from_catalogue]:
                await self.ack(id_)

        try:
            item_to_candidates = await self._get_candidates(items, item_ids)
        except Exception as e:
            await self.rollback(item_ids)
            logging.exception(f"An exception occurred during the getting of the candidates by the all methods. Last exception: {e}")
            metrics.COUNT_METRIC.labels('exception', self.language, "none").inc()
            metrics.COUNT_METRIC.labels('rollback_items_to_queue', self.language, "none").inc(len(item_ids))
            metrics.COUNT_EXCEPTION_METRIC.labels('_get_candidates', self.language).inc()
            return

        st = time.time()
        _ = await asyncio.gather(*[self._process_item(item, item_to_candidates) for item in items])
        logging.info(f"time to process all items {time.time() - st}")

        not_processed_items = await self.get_remains_from_process_queue()

        if not_processed_items:
            logging.warning(
                f"Some items were not processed"
                f", can not obtain the data for these items: {not_processed_items}"
            )
            await self.rollback(not_processed_items)
            metrics.COUNT_METRIC.labels('rollback_items_to_queue', self.language, "none").inc(len(not_processed_items))

    @tenacity.retry(
        reraise=True,
        stop=tenacity.stop_after_attempt(RETRY_ATTEMPTS),
        wait=tenacity.wait_random(min=0, max=2))
    async def get_matches(self, item: dict, candidates: List[dict]):
        time_start = time.time()
        yes, no, unknown = [], [], []

        for candidate in candidates:
            assert candidate["id"] == str(candidate["data"]["id"])

        if self.feature_oc:
            # TODO: load real matchapi ID
            responses, model_info = await self._get_match_many_data(item=item, candidates=candidates, matchapi_id='1')
        else:
            category_responses = await self._get_match_many_data_multi(item=item, candidates=candidates)
            # TODO: we are taking the first of all possible per-category decisions,
            # we can decide to either assume only one possible category for offer, or multiple categories
            # both ways are currently possible, we should decide and then adjust the code accordingly
            # Will fail if `category_responses` is empty, but that is OK
            model_id = list(category_responses.keys())[0]
            responses = category_responses[model_id][0]
            model_info = category_responses[model_id][1]

        for index, candidate in enumerate(candidates):
            response = responses[index]

            compared = {
                "candidate": candidate,
                "details": response["details"],
                "decision": response["match"],
            }

            if response["match"] == Decision.yes.value:
                yes.append(compared)
            elif response["match"] == Decision.no.value:
                no.append(compared)
            elif response["match"] == Decision.unknown.value:
                unknown.append(compared)
            else:
                raise ValueError("Unknown value for match response from MatchAPI.")

        # status prioritization: unpair disabled candidates and prioritize matched candidates by statuses hierarchy
        if self.prioritize_status and len(yes) > 0:
            yes, no, unknown = self._prioritize_status(yes, no, unknown, self.feature_oc)

        # prioritize 'exact' name match
        if len(yes) > 0:
            yes, unknown_additional = self._prioritize_name_match(yes)
            unknown += unknown_additional

        # log info about decision per candidate
        for match_decision in yes + no + unknown:
            candidate = match_decision["candidate"]
            decision = match_decision["decision"]
            details = match_decision["details"]

            cand_source_label = candidate["source"][0] if len(candidate["source"]) == 1 else "both"

            if "namesimilarity=" in details:
                namesimilarity = float(re.search(r"namesimilarity=(\d+(\.)?(\d+)e?-?(\d+)?)", details).group(1))
            else:
                namesimilarity = None

            if "prediction=" in details:
                prediction = float(re.search(r"prediction=(\d+(\.)?(\d+)e?-?(\d+)?)", details).group(1))
            else:
                prediction = None

            if decision == Decision.yes.value:
                # log candidate namesimilarity and predicted probability
                if namesimilarity:
                    metrics.CANDIDATE_NAMESIMILARITY_METRIC.labels(self.language, 'yes').observe(namesimilarity)
                if prediction:
                    metrics.CANDIDATE_PREDICTION_PROBABILITY_METRIC.labels(self.language, 'yes').observe(prediction)

                if "Name match" in details:
                    metrics.COUNT_METRIC_DETAILED.labels('yes_decision_candidate_namesim', self.language, cand_source_label, model_id).inc()
                else:
                    metrics.COUNT_METRIC_DETAILED.labels('yes_decision_candidate', self.language, cand_source_label, model_id).inc()

            elif decision == Decision.no.value:
                # log candidate namesimilarity and predicted probability
                if namesimilarity:
                    metrics.CANDIDATE_NAMESIMILARITY_METRIC.labels(self.language, 'no').observe(namesimilarity)
                if prediction:
                    metrics.CANDIDATE_PREDICTION_PROBABILITY_METRIC.labels(self.language, 'no').observe(prediction)

                if "Keywords mismatch" in details:
                    metrics.COUNT_METRIC_DETAILED.labels('no_decision_candidate_keyword', self.language, cand_source_label, model_id).inc()
                elif "Name attributes mismatch" in details:
                    metrics.COUNT_METRIC_DETAILED.labels('no_decision_candidate_name_attrs', self.language, cand_source_label, model_id).inc()
                elif "Attributes mismatch" in details:
                    metrics.COUNT_METRIC_DETAILED.labels('no_decision_candidate_attrs', self.language, cand_source_label, model_id).inc()
                elif "Numerical unit mismatch" in details:
                    metrics.COUNT_METRIC_DETAILED.labels('no_decision_candidate_unitcheck', self.language, cand_source_label, model_id).inc()
                elif "Name match" in details:
                    metrics.COUNT_METRIC_DETAILED.labels('no_decision_candidate_namesim', self.language, cand_source_label, model_id).inc()
                else:
                    metrics.COUNT_METRIC_DETAILED.labels('no_decision_candidate', self.language, cand_source_label, model_id).inc()

            elif decision == Decision.unknown.value:
                # log candidate namesimilarity and predicted probability
                if namesimilarity:
                    metrics.CANDIDATE_NAMESIMILARITY_METRIC.labels(self.language, 'unknown').observe(namesimilarity)
                if prediction:
                    metrics.CANDIDATE_PREDICTION_PROBABILITY_METRIC.labels(self.language, 'unknown').observe(prediction)

                if "Decision YES, but possible numerical unit mismatch" in details:
                    metrics.COUNT_METRIC_DETAILED.labels('unknown_decision_candidate_unitcheck', self.language, cand_source_label, model_id).inc()
                elif "Decision YES, but other candidate prioritized by status or candidate is disabled" in details:
                    metrics.COUNT_METRIC_DETAILED.labels('unknown_decision_status_prioritization', self.language, cand_source_label, model_id).inc()
                else:
                    metrics.COUNT_METRIC_DETAILED.labels('unknown_decision_candidate', self.language, cand_source_label, model_id).inc()

        if len(yes) == 1:
            final_decision = Decision.yes
            metrics.COUNT_METRIC.labels('yes_decision_final', self.language, model_id).inc()

        elif len(no) == len(candidates):
            final_decision = Decision.no
            metrics.COUNT_METRIC.labels('no_decision_final', self.language, model_id).inc()

        elif len(yes) > 1 or (len(yes) == 0 and len(unknown) > 0):
            final_decision = Decision.unknown
            metrics.COUNT_METRIC.labels('unknown_decision_final', self.language, model_id).inc()

        else:
            raise ValueError("Unknown situation occured.")

        final_candidate = yes[0]["candidate"] if final_decision == Decision.yes else None
        assert final_candidate is None or len(yes) == 1

        time_spent = time.time() - time_start
        metrics.GET_MATCHES_DURATION_SECONDS.labels(
            'get_matches', self.language
        ).observe(time_spent)

        logging.debug("Elapsed time of get_matches %.3fs", time_spent)

        return final_decision, final_candidate, list(itertools.chain(yes, no, unknown)), model_info

    # currently unused
    async def _get_match_data_multi(self, item: dict, candidate: dict) -> dict:
        # the categories should be provided or based on some sort of prediction
        categories = item["categories"]
        matchapi_ids = self.matchapi_manager.get_matchapi_ids_from_categories(categories)
        responses = await asyncio.gather(*[
            self._get_match_data(item, candidate, matchapi_id) for matchapi_id in matchapi_ids
        ])
        # the output of asyncio.gather should be ordered
        # returning {matchapi_id: [responses_for_candidates]}
        return dict(zip(matchapi_ids, responses))

    # currently used only in tests
    @tenacity.retry(
        reraise=True,
        stop=tenacity.stop_after_attempt(RETRY_ATTEMPTS),
        wait=tenacity.wait_random(min=0, max=2))
    async def _get_match_data(self, item: dict, candidate: dict, matchapi_id) -> dict:
        response = {"match": Decision.unknown.value, "details": "No MatchAPIs, default unknown."}
        # basically checking against None
        if matchapi_id in self.matchapi_manager.mapi_id_to_categories:
            async with self.matchapi_manager.get_context_from_id(matchapi_id) as matchapi:
                response = await MatchApiClient(
                    matchapi_service=matchapi
                ).get_match(offer=item, product=candidate)

        return response, self.matchapi_manager.get_model_info_from_id(matchapi_id)

    async def _get_match_many_data_multi(self, item: dict, candidates: List[dict]) -> dict:
        # the categories should be provided or based on some sort of prediction
        categories = item["categories"]
        matchapi_ids = self.matchapi_manager.get_matchapi_ids_from_categories(categories)
        responses = await asyncio.gather(*[
            self._get_match_many_data(item, candidates, matchapi_id) for matchapi_id in matchapi_ids
        ])
        # the output of asyncio.gather is ordered
        # returning {matchapi_id: ([match_results], model_info)}
        return dict(zip(matchapi_ids, responses))

    @tenacity.retry(
        reraise=True,
        stop=tenacity.stop_after_attempt(RETRY_ATTEMPTS),
        wait=tenacity.wait_random(min=0, max=2))
    async def _get_match_many_data(self, item: dict, candidates: List[dict], matchapi_id) -> dict:
        # default response
        response = [{"match": Decision.unknown.value, "details": "No MatchAPIs, default unknown."}] * len(candidates)
        # basically checking against None
        if matchapi_id in self.matchapi_manager.mapi_id_to_categories:
            async with self.matchapi_manager.get_context_from_id(matchapi_id) as matchapi:
                response = await MatchApiClient(
                    matchapi_service=matchapi
                ).get_match_many(offer=item, products=candidates)

        return response, self.matchapi_manager.get_model_info_from_id(matchapi_id)

    @tenacity.retry(
        reraise=True,
        stop=tenacity.stop_after_attempt(RETRY_ATTEMPTS),
        wait=tenacity.wait_random(min=0, max=2))
    async def _get_items_data(self, item_ids: List[str], fields: List[str]) -> List[dict]:
        logging.debug("Request to catalogue service for items info by ids {} ...".format(item_ids))

        async with self.remote_services.get('catalogue').context as catalogue:
            items = await self.catalogue_client(
                catalogue_service=catalogue
            ).get_offers(item_ids, fields)

        return items

    @tenacity.retry(
        reraise=True,
        stop=tenacity.stop_after_attempt(RETRY_ATTEMPTS),
        wait=tenacity.wait_random(min=0, max=2))
    async def _get_candidate_data(self, candidate_ids: List[str], fields: List[str]) -> List[dict]:
        logging.debug("Request to catalogue service for candidates info by ids {} ...".format(candidate_ids))

        async with self.remote_services.get('catalogue').context as catalogue:
            candidates = await self.catalogue_client(
                catalogue_service=catalogue
            ).get_products(product_ids=candidate_ids, fields=fields)

        return candidates

    @tenacity.retry(
        reraise=True,
        stop=tenacity.stop_after_attempt(RETRY_ATTEMPTS),
        wait=tenacity.wait_random(min=0, max=2))
    async def _get_categories_info(self, categories, fields):
        async with self.remote_services.get('catalogue').context as catalogue:
            categories_info_cs2 = await self.catalogue_client(
                catalogue_service=catalogue
            ).get_categories_info(categories=[int(i) for i in categories], fields=fields)

        return categories_info_cs2

    async def update_categories_info(self, categories):
        fields = ['id', 'ean_required', 'unique_names', 'long_tail']
        categories_info_cs2 = await self._get_categories_info(categories, fields)
        categories_info = {
            ci["id"]: {
                "ean_required": ci["ean_required"],
                "unique_names": ci["unique_names"],
                "long_tail": ci["long_tail"],
            }
            for ci in categories_info_cs2
        }
        self.categories_info.update(categories_info)
        return categories_info

    async def get_category_info(self, category_id):
        current_category_data = self.categories_info

        if category_id not in current_category_data:
            category_data = await self.update_categories_info(list(category_id))
            current_category_data = {**self.categories_info, **category_data}

        category_data = current_category_data[category_id]
        logging.info(category_data)

        return category_data

    async def category_info_to_candidates_data(self, candidates: list):
        current_category_data = self.categories_info
        categories_to_update = {c["data"]["category_id"] for c in candidates} - set(self.categories_info)
        if categories_to_update:
            category_data = await self.update_categories_info(list(categories_to_update))
            # just to be sure, the update in update_categories_info sometimes fail for unknown reason
            current_category_data = {**self.categories_info, **category_data}

        for candidate in candidates:
            candidate.update(current_category_data[candidate["data"]["category_id"]])

        return candidates

    @abstractmethod
    def _item_redis_id(self, item: dict) -> str:
        """Get item key for redis.

        This is the value that is stored in UniqueQueue.
        """
