import os
from pathlib import Path
from typing import Optional

import boto3


def list_s3_bucket(
    bucket: str,
    prefix: str,
    endpoint_url: Optional[str] = None,
) -> list:
    endpoint_url = endpoint_url or os.environ.get('S3_ENDPOINT_URL', 'https://s3.heu.cz/')

    client = boto3.client('s3', endpoint_url=endpoint_url)
    result = client.list_objects(Bucket=bucket, Prefix=prefix, Delimiter='/')
    return [r["Prefix"].split(prefix)[1].strip("/") for r in result.get('CommonPrefixes')]



def parse_s3_url(url: str):
    splitted = url.split("/")
    bucket, key, name = splitted[2], "/".join(splitted[3:]), splitted[-1]

    return bucket, key, name


def download_from_s3(
    url: str,
    to_directory: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    destination_path: Optional[str] = None,
) -> str:
    assert to_directory or destination_path, "Specify either destination directory or path"

    endpoint_url = endpoint_url or os.environ.get('MLFLOW_S3_ENDPOINT_URL', 'https://s3.heu.cz/')

    bucket, key, name = parse_s3_url(url)

    # the download could fail if downloading to nonexisting directory
    if to_directory:
        Path(to_directory).mkdir(parents=True, exist_ok=True)

    if not destination_path:
        destination_path = f"{to_directory}/{name}"

    s3 = boto3.resource("s3", endpoint_url=endpoint_url)
    s3.Object(bucket, key).download_file(destination_path)

    return destination_path


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
