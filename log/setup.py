import os
import sys
import logging
import logging.config

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import Config

cfg = Config()


def configure_logging(clear_logs: bool = False):
    # Make sure the logs directory exists
    if not os.path.exists(cfg['logs_dir']):
        os.makedirs(cfg['logs_dir'])

    # Clear logs
    if clear_logs:
        _clear_logs()

    config = cfg['logs_config']
    config = _change_paths(config)
    logging.config.dictConfig(config)


def _clear_logs():
    for file in os.listdir(cfg['logs_dir']):
        os.remove(os.path.join(cfg['logs_dir'], file))


def _change_paths(config: dict) -> dict:
    for handler in config['handlers']:
        if 'filename' in config['handlers'][handler]:
            config['handlers'][handler]['filename'] = os.path.join(cfg['logs_dir'],
                                                                   config['handlers'][handler]['filename'])
    return config
