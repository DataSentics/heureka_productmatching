from utilities.remote_services import get_remote_services
from .candidates import CandidatesProvider
from .candidates_monitor import CandidatesMonitor

from matching_common.clients.provider import FAISS_CANDIDATES_SOURCE, ELASTIC_CANDIDATES_SOURCE


async def get_candidate_provider(
    args_conf, transformer_model_path=None, downloader_params=None, remote_services=None, build_index=True
):
    candidate_provider_config = {}
    if FAISS_CANDIDATES_SOURCE in args_conf.candidates_sources:
        assert transformer_model_path, "Argument `transformer_model_path` not provided"
        candidate_provider_config[FAISS_CANDIDATES_SOURCE] = {
            "input_pmi": args_conf.input_pmi,
            "input_transformer": transformer_model_path,
            "downloader_params": downloader_params,
            "available_data_path": args_conf.input_collector_products,
            "build_index": build_index,
        }

    if ELASTIC_CANDIDATES_SOURCE in args_conf.candidates_sources:
        if not remote_services:
            remote_services = await get_remote_services(['elastic'])

        candidate_provider_config[ELASTIC_CANDIDATES_SOURCE] = {
            "language": "cz",
            "remote_services": remote_services,
        }

    candidate_provider = CandidatesProvider(candidate_provider_config)
    await candidate_provider.init()
    return candidate_provider
