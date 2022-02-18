import os
from pathlib import Path
import random
import sys
import logging
import gzip
import tarfile
import numpy as np
from pathlib import Path

from typing import Optional, Union

# ugly but currently necessary in order to allow import from both Workflow/utilities.py and scripts in SourcesPython
# future unification to some common utilities might solve this issue
try:
    sys.path.append("/app/SourcesPython/utilities")
    from s3_utils import download_from_s3
except ModuleNotFoundError:
    from utilities.s3_utils import download_from_s3


def compress(to: str, path: str):
    with tarfile.open(to, "w:gz") as tar:
        tar.add(path, arcname=os.path.basename(path))


def compress_text_file(to: str, path: str):
    with open(path) as f_in:
        with gzip.open(to, 'wt') as f_out:
            f_out.write(f_in.read())


def set_seed(seed):
    if seed is not None:
        logging.info(f"Seeding {seed}.")
        random.seed(seed)
        np.random.seed(seed)


def process_input(
    input_: Optional[str],
    data_directory: Union[str, Path],
    extract_file: bool = True
) -> Optional[str]:
    if input_ is None:
        return input_

    if input_.startswith("s3://"):
        logging.info(f"Downloading {input_}")
        input_ = download_from_s3(input_, data_directory)

    if input_.endswith(".tar.gz") and extract_file:
        logging.info(f"Untaring {input_}")
        name = input_.split("/")[-1].split(".")[0]

        with tarfile.open(input_) as tar:
            tar.extractall(data_directory)
        input_ = f"{data_directory}/{name}"

    return input_


def process_inputs(
    inputs: Optional[list],
    data_directory: Union[str, Path],
) -> Optional[list]:
    if inputs:
        return [process_input(i, data_directory) for i in inputs]
    else:
        return inputs
