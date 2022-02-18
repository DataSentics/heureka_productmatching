import logging

from collections import defaultdict
from typing import List, Union

from buttstrap.remote_services import RemoteServices
from candy import metrics

from matching_common.clients.elastic_client import ElasticServiceClient
from matching_common.clients.provider import FAISS_CANDIDATES_SOURCE, ELASTIC_CANDIDATES_SOURCE
from matching_common.clients.faiss_client import IndexerApiClient

AVAILABLE_CANDIDATES_SERVICE = {
    FAISS_CANDIDATES_SOURCE: IndexerApiClient,
    ELASTIC_CANDIDATES_SOURCE: ElasticServiceClient
}


class CandidatesProvider:

    def __init__(self, providers_to_use: List[str], remote_services: RemoteServices, language: str, feature_oc=False):
        self.language = language

        self.candidates_providers = []

        for candidates_provider in providers_to_use:
            if candidates_provider in AVAILABLE_CANDIDATES_SERVICE:
                self.candidates_providers.append(
                    AVAILABLE_CANDIDATES_SERVICE[candidates_provider](remote_services, language, feature_oc)
                )

    async def get_candidates(self, items: List[dict], item_ids: List[Union[int, str]], limit: int = 10, **kwargs) -> defaultdict:
        items_candidates = defaultdict(lambda: defaultdict(dict))
        count_exceptions = 0

        for provider in self.candidates_providers:
            logging.debug(f"Gets candidates from {provider}...")
            try:
                if not items_candidates:
                    items_candidates = await provider.get_candidates(items, limit, **kwargs)
                    continue

                other_items_candidates = await provider.get_candidates(items, limit, **kwargs)

                for item_id, other_candidates in other_items_candidates.items():
                    if item_id not in items_candidates:
                        items_candidates[item_id] = other_candidates
                        continue

                    for candidate_id, candidate in other_candidates.items():
                        if candidate_id not in items_candidates[item_id]:
                            items_candidates[item_id][candidate_id] = candidate
                        else:
                            items_candidates[item_id][candidate_id].source.append(provider.get_candidates_source())
            except Exception as e:
                count_exceptions += 1
                logging.exception(f"Exception while _get_candidates_{provider} for item ids {item_ids}.")
                if count_exceptions == len(self.candidates_providers):
                    metrics.COUNT_METRIC.labels('exception', self.language, "none").inc()
                    raise e

        return items_candidates
