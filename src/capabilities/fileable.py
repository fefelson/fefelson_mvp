from abc import ABC, abstractmethod
from typing import Any, List

from ..agents.interfaces import IFileAgent


#################################################################
#################################################################


class Fileable(ABC):
    """Handles loading and saving data to/from files."""
    
    def __init__(self, file_agent: IFileAgent):
        self.file_agent = file_agent    # Manages file I/O operations
        self.filepath: Optional[str] = None  # Path to the file
        self.to_write: bool = False     # Flag to track if save is needed



    @abstractmethod
    def create(self, default):
        pass


    def getFilePath(self):
        return self.filePath


    def getInfo(self):
        return self.info


    @abstractmethod
    def setFilePath(self):
        self.filePath = self._filePath


    def read(self):
        with open(self.filePath) as fileIn:
            temp = json.load(fileIn)
            for key in self.info.keys():
                    self.info[key] = temp[key]


    def write(self):
        try:
            os.makedirs("/".join(self.filePath.split("/")[:-1]))
        except FileExistsError:
            pass

        with open(self.filePath, 'w') as fileOut:
            json.dump(self.info, fileOut)