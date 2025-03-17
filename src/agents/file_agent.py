from abc import ABC, abstractmethod
from typing import Any, Optional

import json 
import pickle

###################################################################
###################################################################


class IFileAgent(ABC):

    @abstractmethod
    def get_ext() -> str:
        raise NotImplementedError
    

    @abstractmethod
    def read(filePath: str) -> Any:
        raise NotImplementedError
    

    @abstractmethod
    def write(filePath: str, fileObject: Any):
        raise NotImplementedError


###################################################################
###################################################################


class JSONAgent(IFileAgent):
    _ext="json"

    @staticmethod
    def get_ext() -> str:
        return JSONAgent._ext
    

    @staticmethod
    def read(filePath: str) -> dict:
        with open(filePath, "r") as fileIn:
            return json.load(fileIn)
        

    @staticmethod
    def write(filePath: str, fileObj: dict):
        with open(filePath, "w") as file:
            json.dump(fileObj, file, indent=4)


###################################################################
###################################################################


class PickleAgent(IFileAgent):
    _ext = "pkl"

    @staticmethod
    def get_ext() -> str:
        return PickleAgent._ext
    

    @staticmethod
    def read(filePath: str) -> Any:
        with open(filePath, "rb") as file:
            return pickle.load(file)
        

    @staticmethod
    def write(filePath: str, fileObj: Any) -> None:
        with open(filePath, "wb") as file:
            pickle.dump(fileObj, file)


#######################################################################
#######################################################################


def get_file_agent(fileType: Optional[str]=None) -> IFileAgent:
    default = "pickle"
    if not fileType:
        fileType = default

    return {"pickle": PickleAgent, "json": JSONAgent}[fileType]