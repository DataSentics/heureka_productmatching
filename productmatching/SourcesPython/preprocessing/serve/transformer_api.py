import os
import fastapi
import pydantic
import numpy as np
from typing import Union, List
from pathlib import Path

from preprocessing.models.sbert_model import SBERTModel

from prometheus_fastapi_instrumentator import Instrumentator

# load model
transformer_model = SBERTModel(os.environ.get("CONFIG__TRANSFORMER_PATH", "/app/data/sbert_model"))


app = fastapi.FastAPI()

Instrumentator().instrument(app).expose(app)


class TransformerRequest(pydantic.BaseModel):
    text: List[str]


@app.get("/ping")
def ping() -> str:
    return "pong"


@app.get("/v1/model-type")
def model_type() -> str:
    return "sbert"


@app.get("/v1/dimension")
def dimension() -> int:
    return transformer_model.get_dimension_sync()


@app.post("/v1/word-vector", response_model=List)
def word_vector(request: TransformerRequest) -> list:
    emb = transformer_model.get_word_vector_sync(request.text)
    return [e.tolist() for e in emb]


@app.post("/v1/sentence-vector", response_model=List)
def sentence_vector(request: TransformerRequest) -> list:
    emb = transformer_model.get_sentence_vector_sync(request.text)
    return [e.tolist() for e in emb]
