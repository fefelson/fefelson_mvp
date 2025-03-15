from abc import ABC, abstractmethod
from typing import Any

import json 
import pickle


###################################################################
###################################################################


class IFileAgent(ABC):
    @abstractmethod
    def read(filePath: str) -> Any:
        pass
    

    @abstractmethod
    def write(filePath: str, fileObject: Any) -> None:
        pass



###################################################################
###################################################################


class JSONAgent(IFileAgent):

    @staticmethod
    def write(filePath: str, fileObj: dict) -> dict:
        with open(filePath, "w") as file:
            json.dump(fileObj, file, indent=4)


    @staticmethod
    def read(filePath: str) -> dict:
        with open(filePath, "r") as fileIn:
            return json.load(fileIn)
        


###################################################################
###################################################################



class PickleAgent(IFileAgent):

    @staticmethod
    def write(filePath: str, fileObj: Any) -> None:
        with open(filePath, "wb") as file:
            pickle.dump(fileObj, file)

    @staticmethod
    def read(filePath: str) -> Any:
        with open(filePath, "rb") as file:
            return pickle.load(file)
