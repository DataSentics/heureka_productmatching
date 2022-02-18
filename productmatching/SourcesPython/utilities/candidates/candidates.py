import logging
import tenacity

from collections import defaultdict
from typing import Dict, List


from utilities.faiss_search import FaissClient
from matching_common.clients.elastic_client import ElasticServiceClient


class CandidatesProvider():
    AVAILABLE_CANDIDATES_SERVICE = {
        'faiss': FaissClient,
        'elastic': ElasticServiceClient
    }

    def __init__(self, providers_to_use: Dict[str, dict]):
        self.providers_to_use = providers_to_use
        self.candidates_providers = {}
        self.providers_w_remote_services = {
            cp for cp, conf in self.providers_to_use.items() if "remote_services" in conf or conf.get("input_transformer", "") == "transformer_api"
        }

    async def init(self):
        for candidates_provider, conf in self.providers_to_use.items():
            if candidates_provider in self.AVAILABLE_CANDIDATES_SERVICE:
                self.candidates_providers[candidates_provider] = self.AVAILABLE_CANDIDATES_SERVICE[candidates_provider](**conf)
                await self.candidates_providers[candidates_provider].init()

    async def close(self):
        for candidates_provider in self.providers_w_remote_services:
            await self.providers_to_use[candidates_provider]["remote_services"].close_all()

    @tenacity.retry(
        reraise=True,
        stop=tenacity.stop_after_attempt(2),
        wait=tenacity.wait_random(min=0, max=2))
    async def get_provider_candidates(self, provider_name, items: List[dict], limit: int = 10, **kwargs) -> defaultdict:
        items_candidates = defaultdict(lambda: defaultdict(dict))
        provider = self.candidates_providers[provider_name]

        try:
            # {item_id: {candidate_id: Candidate, ...}, ...}
            items_candidates = await provider.get_candidates(items, limit, **kwargs)
        except Exception as e:
            logging.exception(f"Exception while _get_candidates {provider} for item ids {[i['id'] for i in items]}. {e.__str__()}")
            return {}

        format_candidates = kwargs.get("format_candidates")
        if format_candidates:
            return self._format_candidates(items_candidates)
        else:
            return items_candidates

    @staticmethod
    def _format_candidates(unformatted_candidates: dict) -> defaultdict:
        if not unformatted_candidates:
            return unformatted_candidates
        # input in format {item_id: {candidate_id: Candidate}}
        # output in format {item_id: [candidate_ids]}
        formatted_candidates = {
            str(item_id): [str(k) for k in candidates_dict.keys()]
            for item_id, candidates_dict in unformatted_candidates.items()
        }

        return formatted_candidates
