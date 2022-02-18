#!/bin/sh
mkdir ${DATA_PATH} -p

for S3_FILE in ${S3_FILES}
do
  if [ "$S3_FILE" != "" ]
  then
    file_name="$(basename -- $S3_FILE)"
    path_to_store="${DATA_PATH}$file_name"

    s3cmd get $S3_FILE $path_to_store \
      --force \
      --host="${S3_HOST}" \
      --host-bucket="${S3_HOST_BUCKET}" \
      --access_key="${S3_ACCESS_KEY}" \
      --secret_key="${S3_SECRET_KEY}"

    tar -xf $path_to_store -C $DATA_PATH || true;
  fi
done

chown root ${DATA_PATH}*
chgrp root ${DATA_PATH}*
