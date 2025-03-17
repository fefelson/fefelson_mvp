from abc import ABC, abstractmethod
from typing import Any, Optional


#################################################################
#################################################################


class Databaseable(ABC):


    def __init__(self, dbAgent: Optional["IDatabaseAgent"]=None):
        self.dbAgent = dbAgent # Handles database operations


    def _set_dbAgent(self, dbAgent: "IDatabaseAgent"):
        self.dbAgent = dbAgent


    @abstractmethod
    def save_to_db(self, data: Any):
        """Saves self.data to the database."""
        raise NotImplementedError   



    @abstractmethod
    def load_from_db(self) -> Any:
        """Loads data from the database into dataclass object using and returns it."""
        raise NotImplementedError  


