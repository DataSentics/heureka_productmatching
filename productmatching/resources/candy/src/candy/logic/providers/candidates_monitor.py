import typing as t

from candy.logic.providers import Candidate
from matching_common.clients.provider import FAISS_CANDIDATES_SOURCE, ELASTIC_CANDIDATES_SOURCE
from candy import metrics


class CandidatesMonitor:
    def __init__(self, sources=[FAISS_CANDIDATES_SOURCE, ELASTIC_CANDIDATES_SOURCE]):
        self.sources = [s for s in sources if s in [FAISS_CANDIDATES_SOURCE, ELASTIC_CANDIDATES_SOURCE]]
        self.similarity_metrics = {
            FAISS_CANDIDATES_SOURCE: "distance",
            ELASTIC_CANDIDATES_SOURCE: "relevance",
        }

    async def monitor_incoming_candidates(self, candidates: t.List[Candidate], language):
        candidates_source = {s: 0 for s in self.sources}
        metrics.CANDIDATES_COUNTS_METRIC.labels(language, 'all').observe(len(candidates))
        for candidate in candidates:
            for cs in candidate.source:
                if cs in self.sources:
                    candidates_source[cs] += 1
                    similarity = candidate.__getattribute__(self.similarity_metrics[cs])
                    if similarity:
                        (
                            metrics
                            .CANDIDATES_SIMILARITY_METRIC
                            .labels(language, self.similarity_metrics[cs])
                            .observe(similarity)
                        )
        for source, n_candidates in candidates_source.items():
            metrics.CANDIDATES_COUNTS_METRIC.labels(language, source).observe(n_candidates)
