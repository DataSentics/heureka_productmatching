import os
import boto3

from pathlib import Path
from typing import Optional


FIBONACCHI = [1, 1, 2, 3, 5, 8]


def download_from_s3(
    url: str,
    to_directory: str,
    endpoint_url: Optional[str] = None,
) -> str:
    endpoint_url = endpoint_url or os.environ.get('S3_ENDPOINT_URL', 'https://s3.heu.cz/')
    splitted = url.split("/")
    bucket, key, name = splitted[2], "/".join(splitted[3:]), splitted[-1]

    # the download could fail if downloading to nonexisting directory
    Path(to_directory).mkdir(parents=True, exist_ok=True)
    s3 = boto3.resource("s3", endpoint_url=endpoint_url)
    path = f"{to_directory}/{name}"
    s3.Object(bucket, key).download_file(path)

    return path


def upload_to_s3(
    url: str,
    path: str,
    endpoint_url: Optional[str] = None,
) -> str:
    endpoint_url = endpoint_url or os.environ.get('S3_ENDPOINT_URL', 'https://s3.heu.cz/')
    splitted = url.split("/")
    bucket, key = splitted[2], "/".join(splitted[3:])

    s3 = boto3.resource("s3", endpoint_url=endpoint_url)
    s3.Object(bucket, key).upload_file(path)

    return path
