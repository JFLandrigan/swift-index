from abc import ABC, abstractmethod

# TODO: add in add and remove methods

class BaseIndex(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def build(self) -> None:
        raise NotImplementedError("This method should be overridden by subclasses")

    @abstractmethod
    def search(self) -> None:
        raise NotImplementedError("This method should be overridden by subclasses")