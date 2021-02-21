"""
Define functions used by this package generally.
"""
from typing import List, Optional
from functools import wraps
import os
import datetime
import time
import logging
import json

def set_tensorflow_gpu():
    import tensorflow as tf # pylint: disable=import-outside-toplevel
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        for physical_device in physical_devices:
            tf.config.experimental.set_memory_growth(physical_device, True)
    except:
        pass

def set_seed(seed: Optional[int] = None):
    if seed is not None:
        import random # pylint: disable=import-outside-toplevel
        import numpy as np # pylint: disable=import-outside-toplevel,redefined-outer-name
        random.seed(seed)
        np.random.seed(seed)

def set_tensorflow_seed(seed: Optional[int] = None):
    if seed is not None:
        set_seed(seed)
        import tensorflow as tf # pylint: disable=import-outside-toplevel
        tf.random.set_seed(seed)

def set_logger(
        log_path: str = 'logs/test.log',
        log_level: int = logging.INFO):

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(process)d] [%(name)s] [%(levelname)s]: %(message)s")

    file_handler = logging.FileHandler(filename=log_path)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(log_level)
    stdout_handler.setFormatter(formatter)

    logging.root.setLevel(log_level)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(stdout_handler)

def set_logging_handler(
        loggers: Optional[List[logging.Logger]] = None,
        log_file_path: str = 'logs/test.log',
        info_level: int = logging.INFO):
    formatter = logging.Formatter(
        "[%(asctime)s] [%(process)d] [%(name)s] [%(levelname)s]: %(message)s")

    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(formatter)

    log_file_handler = logging.FileHandler(filename=log_file_path)
    log_file_handler.setFormatter(formatter)

    if loggers is None:
        logging.basicConfig(handlers=[stdout_handler, log_file_handler], level=info_level)
    else:
        for logger_to_set in loggers:
            logger_to_set.setLevel(info_level)
            logger_to_set.addHandler(stdout_handler)
            logger_to_set.addHandler(log_file_handler)

def dump_json(json_object, file_path):
    with open(file_path, 'w') as _:
        json.dump(
            json_object,
            _,
            ensure_ascii=False,
            indent=4,
            sort_keys=True,
            separators=(',', ': ')
        )

def get_date_str():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

def setup_output_dir(output_dir_path: str, config_dict: dict):
    output_dir_path = os.path.join(output_dir_path, get_date_str())
    os.makedirs(output_dir_path, exist_ok=True)

    dump_json(config_dict, os.path.join(output_dir_path, 'train.json'))

    return output_dir_path

def stop_watch(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        start = time.time()
        result = func(*args, **kargs)
        elapsed_time = time.time() - start
        print('{} は {} 秒かかりました'.format(func.__name__, elapsed_time))

        return result

    return wrapper
