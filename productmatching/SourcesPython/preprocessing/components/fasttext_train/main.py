import os
import sys
import mlflow
import argparse
import logging
import fasttext

from utilities.component import process_input
from utilities.notify import notify
from utilities.logger_to_file import log_to_file_and_terminal

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), force=True)


@notify
def train_fasttext(args):
    corpus_file = args.input_embedding_dataset + "/corpus.txt"
    logging.info(f"Read from {corpus_file}")

    model = fasttext.train_unsupervised(
        corpus_file,
        model=args.model,
        dim=args.dim,
        lr=args.lr,
        ws=args.ws,
        epoch=args.epoch,
        min_count=args.min_count,
        minn=args.minn,
        maxn=args.maxn,
        neg=args.neg,
    )

    model.save_model(f"{args.data_directory}/fasttext.bin")
    mlflow.log_artifact(f"{args.data_directory}/fasttext.bin")

    logging.info("Complete.")
    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-embedding-dataset", required=True)
    parser.add_argument("--data-directory", default="/data")

    parser.add_argument("--model", type=str, default="skipgram")
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--ws", type=int, default=5)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--minn", type=int, default=2)
    parser.add_argument("--maxn", type=int, default=3)
    parser.add_argument("--neg", type=int, default=5)

    args = parser.parse_args()

    args.input_embedding_dataset = process_input(args.input_embedding_dataset, args.data_directory)

    with mlflow.start_run():
        log_to_file_and_terminal(train_fasttext, args)
