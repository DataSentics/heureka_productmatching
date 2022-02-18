import sys
import os
import mlflow
import logging
import traceback
import argparse

from utilities.component import compress, compress_text_file


class Logger(object):
    def __init__(self, std_input, filename):
        self.terminal = std_input
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def log_to_file_and_terminal(fcn_to_run, args: argparse.Namespace, filename: str = "logfile.log"):
    """
    Enable logging to both the terminal and file. In case of failure, full traceback is logged,
    logs are stored and exception is raised. Logs are compressed before logging in mlflow.

    Args:
        fcn_to_run (function): Function to run
        args (argparse.Namespace): Function parameters
        filename (str, optional): Filepath where to write logs. Defaults to "logfile.log".
    """

    # new basicConfig with handlers
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO"),
        force=True,
        handlers=[
            logging.FileHandler(filename, mode="a"),
            logging.StreamHandler(sys.stdout)
            ]
        )
    sys.stdout = Logger(sys.stdout, filename)
    sys.stderr = Logger(sys.stderr, filename)

    # log the traceback if failed
    failed = False
    try:
        fcn_to_run(args)
    except Exception:
        logging.error(traceback.format_exc())
        failed = True

    # compress logfile, tar can be a bit larger for extremely small logfiles but in general it is much smaller than txt
    name = os.path.splitext(filename)[0]
    compressed_log_file = f"{name}.log.gz"
    compress_text_file(compressed_log_file, filename)
    # log to mlflow run (or different sources in future)
    mlflow.log_artifact(compressed_log_file)

    # raise exception if failed, otherwise the workflow would continue
    if failed:
        raise Exception(f"Function '{fcn_to_run.__name__}' failed, check its traceback.")
