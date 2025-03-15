from abc import ABC, abstractmethod
from typing import Any, List, Optional

import os


#################################################################
#################################################################


class Fileable(ABC):
    """Handles loading and saving data to/from files."""
    
    def __init__(self, fileAgent: "IFileAgent"):
        
        self.fileAgent = fileAgent    # Manages file I/O operations
        self.filePath: Optional[str]=None  # Path to the file

    
    @abstractmethod
    def set_file_path(self, filePath: str=None):
        raise NotImplementedError


    def file_exists(self):
        return os.path.exists(self.filePath)

    
    def read_file(self) -> Any:
        return self.fileAgent.read(self.filePath)

    
    def write_file(self, fileableObj: Any) -> None:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.filePath), exist_ok=True)
        self.fileAgent.write(fileableObj)

