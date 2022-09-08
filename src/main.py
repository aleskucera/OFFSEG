#!/usr/bin/env python
import os

import train
from parameter_parser import ParametersImage

MODE = "train"
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_PATH = os.path.join(os.path.dirname(__file__), "..", "models")

PARAMETERS = {
    "train": {"lr": 1e-5,
              "n_epochs": 30,
              "batch_size": 1,
              "dataset_size": 10000,
              "img_size": (320, 512),
              "n_workers": os.cpu_count(),
              "architecture": "fcn_resnet50"},
    "test": {"lr": 1e-5,
             "img_size": (320, 512)}
}


def main():
    # Check macro
    assert MODE in ["train", "test"], "[ERROR] Mode must be 'train' or 'test'"
    assert os.path.exists(DATASET_PATH), "[ERROR] Dataset path must exist"
    assert os.path.exists(MODELS_PATH), "[ERROR] Models path must exist"
    assert len(PARAMETERS[MODE]) > 0, "[ERROR] Parameters must be defined"

    # Read command line and create parameter object
    parameter_parser = ParametersImage(mode=MODE, parameters=PARAMETERS, data_path=DATASET_PATH,
                                       save_path=MODELS_PATH)

    # Train or test the model
    if parameter_parser.mode == "train":
        data = train.train_model(parameter_parser)
        train.plot_training_history(data, save_path=MODELS_PATH)
    elif parameter_parser.mode == "test":
        pass
    else:
        raise Exception("[ERROR] Invalid mode")


if __name__ == '__main__':
    main()
    print("[INFO] Done")
