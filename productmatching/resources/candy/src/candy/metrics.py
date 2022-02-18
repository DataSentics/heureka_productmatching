import time
import numpy as np
from contextlib import contextmanager
from typing import Generator, Union

import prometheus_client.multiprocess
from prometheus_client import Histogram, Counter, Gauge, CollectorRegistry, Summary

APP_NAME = 'Candy'

_REGISTRY = CollectorRegistry()
MULTIPROCESS_REGISTRY = prometheus_client.multiprocess.MultiProcessCollector(_REGISTRY)

COUNT_METRIC = Counter(
    'matching_type_count',
    'Count of processed items by type of output',
    ['type_output', 'language', 'model'],
    registry=_REGISTRY
)

COUNT_METRIC_DETAILED = Counter(
    'matching_decision_detailed_count',
    'Count of matching decisions, detailed',
    ['decision', 'language', 'source', 'model'],
    registry=_REGISTRY
)

COUNT_EXCEPTION_METRIC = Counter(
    'matching_exception_source_count',
    'Count of raised exceptions by source function',
    ['source', 'language'],
    registry=_REGISTRY
)

LATENCY_METRIC = Histogram(
    'api_cycle_duration_seconds',
    'Time spent processing one cycle for selecting items candidates.',
    ['app', 'language'],
    registry=_REGISTRY
)

OFFER_MISSING_FIELDS_COUNT_METRIC = Summary(
    'offer_missing_fields',
    'Number of missing fields of offer to be paired.',
    ['language', 'field'],
    registry=_REGISTRY
)

CANDIDATES_METRIC = Histogram(
    'candidates_search_duration_seconds',
    'Time spent searching for candidates.',
    ['app', 'language', 'source'],
    registry=_REGISTRY
)

CANDIDATES_COUNTS_METRIC = Histogram(
    'candidates_number_source',
    'Number of candidates received from certain source.',
    ['language', 'source'],
    buckets=list(range(0, 21, 1)),
    registry=_REGISTRY
)

CANDIDATES_SIMILARITY_METRIC = Histogram(
    'candidates_similarity',
    'Similarity (relevance/distance) of candidates received.',
    ['language', 'distance_metric'],
    registry=_REGISTRY
)

# this is a basic implementation
CANDIDATE_NAMESIMILARITY_METRIC = Histogram(
    'candidates_namesimilarity',
    'Candidates namesimilarity',
    ['language', 'decision'],  # 'source'
    buckets=list(np.arange(0.0, 1.2, 0.1)),
    registry=_REGISTRY
)

# this is a basic implementation
CANDIDATE_PREDICTION_PROBABILITY_METRIC = Histogram(
    'prediction_probability',
    'Prediction probability',
    ['language', 'decision'],  # 'source'
    buckets=list(np.arange(0.0, 1.1, 0.1)),
    registry=_REGISTRY
)

KAFKA_EXCEPTION_METRIC = Counter(
    'kafka_exception_count',
    'Various errors distinguished by label',
    ['app', 'type', 'topic', 'language'],
    registry=_REGISTRY
)

PROBE_TIME_GAP = Gauge(
    'probe_time_gap',
    'Seconds of passed time between probes.',
    ['app', 'language'],
    registry=_REGISTRY
)

GET_MATCHES_DURATION_SECONDS = Histogram(
    "get_matches_duration_seconds", "Time spent processing matches",
    labelnames=["app", "language"],
    registry=_REGISTRY
)


@contextmanager
def measure(metric: Union[Histogram, Counter], **kwargs) -> Generator:
    def _save_metric_value(_start_time: float):
        resp_time = time.time() - _start_time
        metric.labels(app=APP_NAME, **kwargs).observe(resp_time)

    start_time = time.time()
    yield
    _save_metric_value(start_time)
