import os
import argparse
import mlflow
import logging
from comparison.utils import ModelComparator
from utilities.notify import notify

from utilities.component import process_input

@notify
def main(args):

    output_path = os.path.join(args.data_directory, args.output_filename)

    comparator = ModelComparator()
    # load excels
    comparator.load(args.excel_path, args.validation_excel_path)
    # create comparison excel and save it
    comparator.create_comparison_excel(output_path)

    mlflow.log_artifact(output_path)



if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), force=True)
    parser = argparse.ArgumentParser()

    parser.add_argument("--excel-path", required=True)
    parser.add_argument("--validation-excel-path", default=None)
    parser.add_argument("--data-directory", default="/data")
    parser.add_argument("--output-filename", default="comparison_excel.xlsx")
    args = parser.parse_args()

    args.excel_path = process_input(args.excel_path, args.data_directory)

    # excels would overwrite themselves if samely named, use another directory to store the second 
    data_directory_for_comparison = args.data_directory + "_compare"
    args.validation_excel_path = process_input(args.validation_excel_path, data_directory_for_comparison)

    logging.info(args)

    with mlflow.start_run():
        main(args)