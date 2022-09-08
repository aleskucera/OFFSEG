import os
import yaml
import logging
import logging.config

LOGS_DIR = os.path.join(os.path.dirname(__file__), 'logs')
CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config.yaml')


def configure_logging(clear_logs: bool = False):
    # Make sure the logs directory exists
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
        
    # Clear logs
    if clear_logs:
        _clear_logs()

    # Load logging configuration
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f.read())
        config = _change_paths(config)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)


def _clear_logs():
    for file in os.listdir(LOGS_DIR):
        os.remove(os.path.join(LOGS_DIR, file))


def _change_paths(config: dict) -> dict:
    for handler in config['handlers']:
        if 'filename' in config['handlers'][handler]:
            config['handlers'][handler]['filename'] = os.path.join(LOGS_DIR, config['handlers'][handler]['filename'])
    return config
