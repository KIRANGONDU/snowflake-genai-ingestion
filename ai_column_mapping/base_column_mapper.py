from abc import ABC, abstractmethod

class BaseColumnMapper(ABC):

    @abstractmethod
    def map_columns(self, source_columns: list, canonical_schema: dict) -> dict:
        pass
