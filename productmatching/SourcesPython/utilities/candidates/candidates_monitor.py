import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import typing as t
from collections import defaultdict

from matching_common.clients.candidate import Candidate
from matching_common.clients.provider import FAISS_CANDIDATES_SOURCE, ELASTIC_CANDIDATES_SOURCE

CANDIDATES_SOURCES = [FAISS_CANDIDATES_SOURCE, ELASTIC_CANDIDATES_SOURCE]


class CandidatesMonitor:
    def __init__(self, sources: t.List[str]):
        self.sources = [s for s in sources if s in [FAISS_CANDIDATES_SOURCE, ELASTIC_CANDIDATES_SOURCE]]
        self.similarity_data = {s: [] for s in self.sources}
        self.candidates_unique = defaultdict(set)
        self.n_candidates_per_source = {s: 0 for s in self.sources}
        self.rank_paired_per_item_source = defaultdict(dict)
        self.n_processed_total = 0
        self.similarity_metrics = {
            FAISS_CANDIDATES_SOURCE: "distance",
            ELASTIC_CANDIDATES_SOURCE: "relevance",
        }
        self.pdf_source_statistics = pd.DataFrame()

    def monitor_incoming_candidates(self, candidates: t.List[Candidate], item_id: int, id_paired: int):
        # `candidates` should be a list of Candidate objects obtained from one item/offer from one source
        # `id_paired` is id of currently paired product for the offer under `item_id`
        self.n_processed_total += 1
        for source in CANDIDATES_SOURCES:
            source_candidates = [c for c in candidates if all([s == source for s in c.source])]
            if source_candidates:
                # sorting the candidates by similarity
                source_candidates = sorted(
                    source_candidates,
                    key=lambda c: c.relevance if c.relevance else c.distance
                )
                c_ids = [c.id for c in source_candidates]

                self.candidates_unique[item_id] |= set(c_ids)
                # rank of paired product among candidates, np.inf when it is not present
                try:
                    paired_idx = c_ids.index(str(id_paired))
                except ValueError:
                    paired_idx = np.inf

                # storing the data on rank of paired product among candidates
                self.rank_paired_per_item_source[item_id][source] = paired_idx
                for candidate in source_candidates:
                    # storing the data on number of candidates from the `source` and their similarity
                    self.n_candidates_per_source[source] += 1
                    similarity = candidate.__getattribute__(self.similarity_metrics[source])
                    if similarity:
                        self.similarity_data[source].append(similarity)

    def produce_candidate_statistics(self, create_plot: bool):
        n_candidates_total = sum(self.n_candidates_per_source.values())
        n_candidates_unique = sum(len(c) for c in self.candidates_unique.values())

        # find the highest rank among candidates from both sources for each offer
        min_ranks_total = defaultdict(int)
        ranks_source = {cs: defaultdict(int) for cs in self.sources}
        for id, ranks in self.rank_paired_per_item_source.items():
            min_rank = min(i for i in ranks.values())
            if min_rank == np.inf:
                min_rank = "not_present"
            min_ranks_total[min_rank] += 1
            # ranks for sources
            for cs, rank in ranks.items():
                if rank == np.inf:
                    ranks_source[cs]["not_present"] += 1
                else:
                    ranks_source[cs][rank] += 1

        # quantiles fof similarity distributions to be reported
        quants = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        results = {}
        results['n_candidates_total'] = n_candidates_total
        results['n_candidates_unique'] = n_candidates_unique
        results['n_candidates_avg'] = n_candidates_total / self.n_processed_total if self.n_processed_total > 0 else 0
        # number of times the paired product was listed at index `idx` among candidates
        # "not_present" when it is not present
        results['n_paired_position'] = {idx: n for idx, n in min_ranks_total.items()}
        # assuming we try to get candidates from all available sources for each offer

        for cs in self.sources:
            results[cs] = {}
            results[cs]['n_candidates_total'] = self.n_candidates_per_source[cs]
            results[cs]['n_candidates_avg'] = self.n_candidates_per_source[cs] / self.n_processed_total if self.n_processed_total > 0 else 0
            results[cs]['n_paired_position'] = {idx: n for idx, n in ranks_source[cs].items()}
            if self.n_candidates_per_source[cs] > 0:
                results[cs]['similarity_quantiles'] = {
                    # str: float
                    f"quantile_{q}": qq for q, qq in zip(quants, np.quantile(self.similarity_data[cs], quants))
                }
            else:
                results[cs]['similarity_quantiles'] = {f"quantile_{q}": 0 for q in quants}

        if create_plot and self.n_processed_total and n_candidates_total:
            # TODO: it may turn out that distance and relvance have distributions on different scale
            # in such case, plot should be somehow divided
            maxlen = max(len(v) for v in self.similarity_data.values())
            plot_data = {
                cs: sim_list + [None for i in range(maxlen - len(sim_list))]
                for cs, sim_list in self.similarity_data.items()
                if self.n_candidates_per_source[cs] > 0
            }
            pdf = pd.DataFrame(plot_data)
            pdf = pdf.melt(value_vars=plot_data.keys())
            pdf.columns = ['source', 'similarity']
            with sns.axes_style("darkgrid"):
                results['similarity_plot'] = sns.violinplot(x='similarity', y='source', data=pdf).get_figure()
        else:
            results['similarity_plot'] = None

        self.results = results
        self.statistics_to_df()

    def statistics_to_df(self):
        # turn the monitoruing results to pandas df, with columns for source, stratistic and its value
        rows = [
            ('all', 'n_candidates_total', self.results['n_candidates_total']),
            ('all', 'n_candidates_avg', self.results['n_candidates_avg']),
        ]
        for idx, n in self.results['n_paired_position'].items():
            rows.append(('all', f'n_paired_position_{idx}', n))

        for source in self.sources:
            source_data = self.results[source]
            rows.extend([
                (source, 'n_candidates_total', source_data['n_candidates_total']),
                (source, 'n_candidates_avg', source_data['n_candidates_avg']),
            ])
            for q, qv in source_data['similarity_quantiles'].items():
                rows.append((source, q, qv))
            for idx, n in source_data['n_paired_position'].items():
                rows.append((source, f'n_paired_position_{idx}', n))
        self.pdf_source_statistics = pd.DataFrame(rows, columns=['source', 'statistic', 'value'])

    def log_metrics_mlflow(self):
        for i, row in self.pdf_source_statistics.iterrows():
            mlflow.log_metric('_'.join([row[0], row[1]]), row[2])
