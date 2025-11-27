from numpy.typing import NDArray
from abc import ABC


class BaseModel(ABC):
    def _trained(self):
        raise NotImplementedError()
    
    @property
    def trained(self):
        return self._trained()
    
    def train(self, x: NDArray, y: NDArray, epochs: int, **kwargs):
        raise NotImplementedError()
    
    def predict(self, x: NDArray, **kwargs):
        raise NotImplementedError()