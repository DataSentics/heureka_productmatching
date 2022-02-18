import os
import re
import nltk
import mlflow
import argparse
import logging
import numpy as np

from utilities.component import process_input, compress
from utilities.preprocessing import Pipeline
from utilities.normalize.regexes import get_units
from utilities.loader import Corpus
from utilities.logger_to_file import log_to_file_and_terminal


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), force=True)


def pmi(args):

    def get_docs(args):
        with open(args.input_embedding_dataset + "/corpus.txt", "r") as f:
            for line in f.readlines():
                yield line.split(" ")

    units = get_units()
    NUM_UNIT = units["NUM_UNIT"]

    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder = nltk.collocations.BigramCollocationFinder.from_documents(get_docs(args))

    # Remove less frequent words
    finder.apply_freq_filter(args.min_token_frequency)

    # Remove very short words
    finder.apply_word_filter(lambda word: len(word) < args.min_token_length)

    # Remove words with units, e.g. 100g, 10kg, etc.
    num_unit_pattern = re.compile(NUM_UNIT)
    finder.apply_word_filter(lambda word: num_unit_pattern.match(word))

    # remove words with "a lot" of numbers, e.g. product codes
    finder.apply_word_filter(lambda word: sum(c.isalpha() for c in word)/sum(c.isdigit() for c in word) < 3 if sum(c.isdigit() for c in word) else False)

    # TODO: use when experimenting with n-grams, n=2,3, ... instead of l.53
    # # calculate scores
    # scores = [score for bigram, score in finder.score_ngrams(bigram_measures.pmi)]
    # # calculate threshold as quantile
    # thr = np.quantile(scores, 0.75)
    # mlflow.log_metric("Threshold score", thr)
    # best = finder.above_score(bigram_measures.pmi, thr)

    best = finder.above_score(bigram_measures.pmi, args.min_pmi_score)
    pmi_file = args.data_directory + "/pmi.txt"

    for seq in best:
        line = " ".join(seq).strip()

        if not line:
            continue
        Corpus.save(pmi_file, line)

    mlflow.log_metric("n", len(list(best)))

    # TODO this is not ideal, the Pipeline might need other arguments (? maybe)
    pipeline = Pipeline.create(pmi_file=pmi_file)

    pmi_embedding_dataset = args.data_directory + "/pmi_embedding_dataset/corpus.txt"

    for line in get_docs(args):
        Corpus.save(pmi_embedding_dataset, pipeline(" ".join(line)))

    Corpus.close()

    with open(pmi_file, "a"):
        pass

    mlflow.log_artifact(pmi_file)

    tar_file = args.data_directory + "/pmi_embedding_dataset.tar.gz"
    compress(tar_file, args.data_directory + "/pmi_embedding_dataset")
    mlflow.log_artifact(tar_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-embedding-dataset", required=True)
    parser.add_argument("--data-directory", default="/data")

    parser.add_argument("--min-token-frequency", type=int, default=22)
    parser.add_argument("--min-token-length", type=int, default=4)
    parser.add_argument("--min-pmi-score", type=int, default=10000)

    args = parser.parse_args()

    args.input_embedding_dataset = process_input(args.input_embedding_dataset, args.data_directory)

    with mlflow.start_run():
        pmi(args)