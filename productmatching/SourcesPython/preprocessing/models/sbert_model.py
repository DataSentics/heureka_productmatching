import os
import logging
from typing import Union
from utilities.component import process_input

from sentence_transformers import SentenceTransformer, util


class SBERTModel:
    def __init__(self, model_path: str):
        # load the model
        self.model = self.load_model(model_path)
        self.vector_dim = self.model.get_sentence_embedding_dimension()

    async def init(self):
        pass

    async def close(self):
        pass

    def load_model(self, model_path: str):
        if not model_path or not os.path.isdir(model_path):
            # download model from S3, used in tests and in local mode for the first time or when no path specified
            data_directory = os.path.dirname(model_path) if model_path else "/data"

            logging.info("Downloading sBERT model from S3")
            model_path = process_input(
                os.environ.get(
                    "SBERT_MODEL_ADDRESS",
                    "s3://matchapi-data-cz/external_models/sbert_model.tar.gz"
                ),
                data_directory
            )
        return SentenceTransformer(model_path)

    async def get_dimension(self):
        return self.vector_dim

    # for transformer API
    def get_dimension_sync(self):
        return self.vector_dim

    async def get_sentence_vector(self, sentence: Union[str, list], show_progress_bar: bool = False):
        return self.model.encode(sentence, show_progress_bar=show_progress_bar)

    # for transformer API
    def get_sentence_vector_sync(self, sentence: Union[str, list], show_progress_bar: bool = False):
        return self.model.encode(sentence, show_progress_bar=show_progress_bar)

    async def get_word_vector(self, word: Union[str, list], show_progress_bar: bool = False):
        # we could get subword embeddings by
        # self.model.encode(sentence, output_value='token_embeddings'),
        # their average equals encoding whole one-word sentence
        vec = await self.get_sentence_vector(word, show_progress_bar=show_progress_bar)
        return vec

    # for transformer API
    def get_word_vector_sync(self, word: Union[str, list], show_progress_bar: bool = False):
        # we could get subword embeddings by
        # self.model.encode(sentence, output_value='token_embeddings'), 
        # their average equals encoding whole one-word sentence
        return self.get_sentence_vector_sync(word, show_progress_bar=show_progress_bar)
