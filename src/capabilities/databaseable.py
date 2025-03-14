from abc import ABC, abstractmethod
from typing import Any, List

from ..agents import IDatabaseAgent


#################################################################
#################################################################


class Databasable(ABC):

    def __init__(self, dbAgent: IDatabaseAgent):
        self.dbAgent = dbAgent  # Handles database operations


    @abstractmethod
    def save_to_db(self) -> None:
        """Saves self.data to the database."""
        print(f"Databaseable.save_to_db called with data: {self.data}")
        

    @abstractmethod
    def load_from_db(self) -> List[Any]:
        """Loads data from the database into dataclass object using and returns it."""
        print(f"Databaseable.load_from_db called ")
