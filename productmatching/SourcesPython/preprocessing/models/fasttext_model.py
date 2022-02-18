import fasttext
from typing import Union


class FastTextModel:
    def __init__(self, model_path: str):
        self.model = self.load_model(model_path)
        self.vector_dim = self.model.get_dimension()

    async def init(self):
        pass

    async def close(self):
        pass

    def load_model(self, model_path: str):
        return fasttext.load_model(model_path)

    async def get_dimension(self):
        return self.vector_dim

    async def get_sentence_vector(self, sentence: Union[str, list], show_progress_bar: bool = False):
        # enable computation in batch mode and for individual sentence
        if isinstance(sentence, str):
            # only one vector
            return self.model.get_sentence_vector(sentence)
        else:
            # list of vectors
            return [
                self.model.get_sentence_vector(sent)
                for sent in sentence
            ]

    async def get_word_vector(self, word: Union[str, list], show_progress_bar: bool = False):
        # receive one sentence or batch of sentences
        if isinstance(word, str):
            # only one vector
            return self.model.get_word_vector(word)
        else:
            # list of vectors
            return [
                self.model.get_word_vector(w)
                for w in word
            ]
