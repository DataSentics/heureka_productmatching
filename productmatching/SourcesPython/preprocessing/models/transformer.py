import warnings
from itertools import chain
from typing import Union, List
from pathlib import Path

from preprocessing.models.sbert_model import SBERTModel
from preprocessing.models.fasttext_model import FastTextModel
from utilities.preprocessing import Pipeline
# TransformerModelApi can be imported later in script if it should be used
# do not import it here to avoid installation problems in GCP

AVAILABLE_TRANSFORMERS = {
    "sbert": SBERTModel,
    "fasttext": FastTextModel,
}


class TransformerModel:

    def __init__(self, model_path: Union[str, Path] = None):
        # if no path specified, use sbert and download it from specified location
        # do not use 'sbert' as default to prevent unwanted masking incorrect model path by default sbert
        if model_path:
            model_path = str(model_path)
            self.model_type = self.get_model_type_from_path(model_path)
        else:
            self.model_type = "sbert"
            warnings.warn(f"No model path specified, default {self.model_type} will be downloaded and used")

        if self.model_type == "transformer_api":
            from preprocessing.models.transformer_api import TransformerModelApi
            AVAILABLE_TRANSFORMERS["transformer_api"] = TransformerModelApi

        self.model = AVAILABLE_TRANSFORMERS[self.model_type](model_path)

    async def init(self):
        await self.model.init()

    async def close(self):
        await self.model.close()

    async def close_all(self):
        """To keep compatibility with closing used in CandidatesProvider class"""
        await self.model.close()

    async def get_dimension(self):
        dim = await self.model.get_dimension()
        return dim

    async def get_word_vector(self, word: Union[str, list], show_progress_bar: bool = False):
        vec = await self.model.get_word_vector(word, show_progress_bar=show_progress_bar)
        return vec

    async def get_sentence_vector(self, sentence: Union[str, list], show_progress_bar: bool = False):
        vec = await self.model.get_sentence_vector(sentence, show_progress_bar=show_progress_bar)
        return vec

    def get_model_type_from_path(self, model_path: str):
        if "fasttext.bin" in model_path:
            return "fasttext"
        elif "http:/" in model_path:
            return "transformer_api"
        elif "bert" in model_path:
            return "sbert"
        else:
            raise Exception(f"Incorrect transformer model path {model_path}")

    async def sentences_to_wbyw_embeddings(self, sentences: List[str], pipeline: Pipeline = None) -> list:
        """
        The 'wbyw'stands for 'word by word'.
        Currently, a sentence (product/offer name) embeddings for NN are just lists of embeddings of individual words.
        This function aims to convert a huge batch of sentences to this form at once in order to save some CPU time.
        """
        def embedding_or_none(nomalized_sentence: list, word_embedding_map: dict):
            if nomalized_sentence == [""]:
                return None
            return [word_embedding_map[s] for s in nomalized_sentence]

        if pipeline:
            normalized_sentences = [pipeline(s).split(" ") for s in sentences]
        else:
            normalized_sentences = [s.split(" ") for s in sentences]

        # we don't want to embedd empty strings, but we want to keep it in original list
        to_embedd = list(set(chain(*[s for s in normalized_sentences if s])))
        embeddings = await self.get_sentence_vector(to_embedd)
        word_embedding_map = dict(zip(to_embedd, embeddings))

        embedded_sentences = [embedding_or_none(ns, word_embedding_map) for ns in normalized_sentences]
        return embedded_sentences
