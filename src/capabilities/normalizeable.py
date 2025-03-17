from abc import ABC, abstractmethod
from typing import Any, Optional


##################################################################
##################################################################



class Normalizable(ABC):
    """Handles normalizg web data."""
    
    def __init__(self, normalAgent: Optional["INormalAgent"]=None):
        self.normalAgent = normalAgent  


    def _set_normal_agent(self, normalAgent: "INormalAgent"):
        self.normalAgent = normalAgent 



    @abstractmethod
    def normalize(self, webData: dict) -> Any:
        pass