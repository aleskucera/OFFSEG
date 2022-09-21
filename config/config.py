import os
import yaml

# Add here configuration parameters which keys need to be converted to a integer
NAMES_FOR_CONVERSION = ['dataset_labels',
                        'dataset_color_map',
                        'rellis_label_map',
                        'rugd_label_map',
                        'rugd_color_map']

# Configuration files path
logs_config = os.path.join(os.path.dirname(__file__), 'logs.yaml')
dataset_config = os.path.join(os.path.dirname(__file__), 'dataset.yaml')
training_config = os.path.join(os.path.dirname(__file__), 'training.yaml')
testing_config = os.path.join(os.path.dirname(__file__), 'testing.yaml')
project_structure = os.path.join(os.path.dirname(__file__), 'project_structure.yaml')

# Dataset configuration
rellis_config = os.path.join(os.path.dirname(__file__), 'rellis_dataset.yaml')
rugd_config = os.path.join(os.path.dirname(__file__), 'rugd_dataset.yaml')
cityscapes_config = os.path.join(os.path.dirname(__file__), 'cityscapes_dataset.yaml')

root_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))


class Config(object):
    def __init__(self):
        assert os.path.exists(logs_config), f"Logs_config '{logs_config}' does not exist"
        assert os.path.exists(dataset_config), f"Dataset_config '{dataset_config}' does not exist"
        assert os.path.exists(training_config), f"Training_config '{training_config}' does not exist"
        assert os.path.exists(testing_config), f"Testing_config '{testing_config}' does not exist"
        assert os.path.exists(project_structure), f"Project_structure '{project_structure}' does not exist"

        self.names_for_conversion = NAMES_FOR_CONVERSION

        # load constants from yaml file
        self.logs_config = self.load_config(logs_config)
        self.dataset_config = self.load_config(dataset_config)
        self.training_config = self.load_config(training_config)
        self.testing_config = self.load_config(testing_config)
        self.project_structure = self.load_config(project_structure)
        self.rellis_config = self.load_config(rellis_config)
        self.rugd_config = self.load_config(rugd_config)
        self.cityscapes_config = self.load_config(cityscapes_config)

        # check if project structure exists, set paths to absolute paths and check if they exist
        for key, value in self.project_structure.items():
            absolute_path = self.absolute_path(value)
            assert os.path.exists(absolute_path), f"Path '{absolute_path}' does not exist"
            self.project_structure[key] = absolute_path

    @staticmethod
    def absolute_path(path):
        return os.path.realpath(os.path.join(os.path.dirname(__file__), '..', path))

    @staticmethod
    def load_config(path: str):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return data

    @staticmethod
    def integer_keys(dictionary):
        return {int(k): v for k, v in dictionary.items()}

    def __getitem__(self, item):
        configs = [self.dataset_config,
                   self.training_config,
                   self.testing_config,
                   self.logs_config,
                   self.project_structure,
                   self.rellis_config,
                   self.rugd_config,
                   self.cityscapes_config]

        # check if item is in any of the configs
        for cfg in configs:
            if cfg is not None and item in cfg:

                # check if item is in the list of names that need to be converted to integers
                if item in self.names_for_conversion:
                    return self.integer_keys(cfg[item])
                else:
                    return cfg[item]


if __name__ == '__main__':
    c = Config()
    print(os.path.join(c['root_dir'], 'data'))
