from typing import Union
import argparse

from utilities.component import process_input

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str_or_none(st):
    if st and str(st).lower() == "none":
        return None

    return st


def set_at_args(args: argparse.Namespace, field: str, data_directory, separator: str = '@@@'):
    """ For each "arg" in `separator` separated string in args.field, with format key=value:
        add key to args with value value """
    for key, value in convert_at_args(args.__dict__[field], data_directory).items():
        args.__setattr__(key, value)

    # add `data_directory` to the `field`
    val = separator.join([f'data_directory={data_directory}', args.__dict__[field]])
    args.__setattr__(field, val)

    return args

def convert_at_args(tok_norm_args: Union[str, dict], data_directory:str = None, separator:str = '@@@'):
    """ Convert `tok_norm_args` to dict.
     Each "arg" in `separator` separated string in args.field, with format key=value """
    d = {}
    if not tok_norm_args:
        return {}
    if isinstance(tok_norm_args, dict):
        return tok_norm_args

    for item in tok_norm_args.strip().split(separator):
        key, value = item.split('=', 1)
        if data_directory:
            value = process_input(value, data_directory)
        d[key] = value
    return d


