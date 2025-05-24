from abc import ABC, abstractmethod

class BaseDataProcessor(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        self.x_full_dataset = None
        self.y_full_dataset = None

    @abstractmethod
    def process_file(self, filename):
        pass