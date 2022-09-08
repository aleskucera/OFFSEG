import torch
import logging
import argparse


class ParametersImage(object):
    def __init__(self, mode: str = 'train', parameters: dict = None, data_path: str = None, save_path: str = None):
        # Create logger
        logger = logging.getLogger(__name__)

        # parse arguments from command line
        assert mode in ['train', 'test'], "[ERROR] Mode must be 'train' or 'test'"
        assert parameters is not None, "[ERROR] Parameters must be provided"
        assert save_path is not None, "[ERROR] Save_path must be provided"

        args = self._parse_args(parameters, mode)

        # set the parameters
        self.mode = mode
        self.data_path = data_path
        self.save_path = save_path
        self.img_size = args.img_size
        if mode == "train":
            self.lr = args.lr
            self.n_epochs = args.n_epochs
            self.plot_path = args.plot_path
            self.n_workers = args.n_workers
            self.batch_size = args.batch_size
            self.architecture = args.architecture
            self.dataset_size = args.dataset_size
        elif mode == "test":
            self.dataset = args.dataset
        else:
            raise Exception("Invalid type")

        # set the device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        logger.debug(f"Parameters: {self.__dict__}")

    @staticmethod
    def _parse_args(parameters: dict, mode: str = 'train') -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        for key, value in parameters[mode].items():
            parser.add_argument(f"--{key}", type=type(value), default=value)
        return parser.parse_args()