from enum import Enum

REGISTERED_DATASET = ["highD", "inD", "rounD", "exiD", "uniD"]

class LevelXParser(object):
    def __init__(self, dataset: str):
        if dataset not in REGISTERED_DATASET:
            raise KeyError(f"{dataset} is not an available LevelX-series dataset.")

        self.dataset = dataset