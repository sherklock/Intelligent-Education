from abc import ABC, abstractmethod

class AbstractStrategy(ABC):
    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError

    @abstractmethod
    def adaptest_select(self, model, adaptest_data, it=None, test_length=None):
        raise NotImplementedError