from abc import ABC, abstractmethod

class AbstractProcessData(ABC):
    instance = None
    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance

    @abstractmethod
    def __init__(self, *args, **kwargs):
        self.x_full_dataset = None
        self.y_full_dataset = None

    @staticmethod
    @abstractmethod
    def process_file(self, filename):
        pass