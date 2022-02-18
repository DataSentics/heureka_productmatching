import asyncio
import itertools
import tenacity

from typing import Union
from pathlib import Path

from aiohttp import ClientResponse, ClientTimeout
from llconfig import Config
from llconfig.converters import json as js

from buttstrap.remote_services import RemoteServices
from preprocessing.models.sbert_model import SBERTModel
from preprocessing.models.fasttext_model import FastTextModel


class TransformerModelApi:

    DIMENSION_ENDPOINT = "v1/dimension"
    WORD_ENDPOINT = "v1/word-vector"
    SENTENCE_ENDPOINT = "v1/sentence-vector"
    # following thresholds depend on number and resources of running transformer-api pods
    # used limits were tested and used for 5 running pods with limit of 6 CPU and 5GB RAM
    # with these values, up to 4 simultaneously running workflows can be deployed without timeouts
    # number of items sent in one batch (one request)
    MAX_BATCH_ITEMS = 20
    # number of batches sent at once using asyncio gather
    MAX_ASYNC_REQUESTS = 8

    def __init__(self, model_path: Union[str, Path] = None):
        self.api_url = model_path

    async def init(self):
        conf = Config()
        conf.init('TRANSFORMER', js)
        conf["TRANSFORMER"] = {
            "address": self.api_url,
            "timeout": ClientTimeout(total=10000000)
        }
        params = {"rest": ["transformer"], "conf": conf}

        remote_services = RemoteServices(**params)

        await remote_services.init()
        self.remote_services = remote_services

    async def close(self):
        await self.remote_services.close_all()

    async def _process_response(self,
                                response: ClientResponse,
                                response_type: str = "json") -> Union[str, dict, ClientResponse]:
        if response.status == 200:
            if response_type == "json":
                result = await response.json()
            elif response_type == "text":
                result = await response.text()
            else:
                result = response

            return result

    async def get_dimension(self):
        async with self.remote_services.get('transformer').context as transformer:
            response = await transformer.request('GET', self.DIMENSION_ENDPOINT)

            response = await self._process_response(response, "text")

            return int(response)

    @tenacity.retry(
        reraise=False,
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_random(min=0, max=2))
    async def request_batch(self, json: dict, endpoint):
        """Receives text to be embbeded, sends the request, processes response and returns it."""
        async with self.remote_services.get('transformer').context as transformer:
            response = await transformer.request('POST', endpoint, json=json)
            response = await self._process_response(response, "json")

        return response

    async def get_embeddings(self, words: Union[str, list], endpoint: str):
        """Transforms input words (either str or list) into batches of requests, calls the api with specified endpoint and puts it back together."""
        # transform input to list
        text = self.to_list(words)

        # divide in batches of maximal possible legth and parse it into message dict
        data_json_list = [
            self.text_to_msg(
                text[start_ind: min(start_ind+self.MAX_BATCH_ITEMS, len(text))]
            )
            for start_ind in range(0, len(text), self.MAX_BATCH_ITEMS)
        ]
        # process batches, call specified number of requests at once
        results = []
        for batch_start_ind in range(0, len(data_json_list), self.MAX_ASYNC_REQUESTS):
            coros = [
                self.request_batch(
                    data_json,
                    endpoint,
                )
                for data_json in data_json_list[
                    batch_start_ind: min(batch_start_ind+self.MAX_ASYNC_REQUESTS, len(data_json_list))
                ]
            ]
            response = await asyncio.gather(*coros)
            # add to all results
            results.extend(itertools.chain(*response))

        if isinstance(words, str):
            return results[0]
        else:
            return results

    async def get_word_vector(self, word: Union[str, list], **kwargs):
        vec = await self.get_embeddings(word, self.WORD_ENDPOINT)
        return vec

    async def get_sentence_vector(self, word: Union[str, list], **kwargs):
        vec = await self.get_embeddings(word, self.SENTENCE_ENDPOINT)
        return vec

    @staticmethod
    def to_list(text: Union[str, list]):
        if isinstance(text, str):
            text = [text]
        return text

    @staticmethod
    def text_to_msg(text: Union[str, list]):
        if isinstance(text, str):
            data_json = {"text": [text]}
        else:
            data_json = {"text": text}

        return data_json
