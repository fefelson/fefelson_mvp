from typing import Dict, Any
from threading import Lock

from ..capabilities import Fileable
from ..agents import IFileAgent, JSONAgent


##################################################################
##################################################################



class LeagueConfig(Fileable):
    _config_lock = Lock()  # Prevent race conditions in multi-threaded environments

    def __init__(self, 
                 leagueId: str,
                 fileAgent: IFileAgent = JSONAgent
                 ):
        Fileable.__init__(self, fileAgent)
        
        self.set_file_path()
        self.config: Dict[str, Any] = self._load_config(leagueId)


    def _load_config(self, leagueId: str) -> Dict[str, Any]:
        """Loads the full config file and extracts the league-specific settings."""
        fullConfig = self.read_file()
        return fullConfig.get(leagueId, {})  # Get only this league’s config
        

    def _write_config(self) -> None:
        """Writes only the updated league's config without affecting others."""
        with self._config_lock:
            fullConfig = self.read(self.filePath)
            fullConfig[self.league_id] = self.config  # Update only this league’s config
            self.write_file(fullConfig)  # Save changes



    def get(self, key: str, default: Any = None) -> Any:
        """Retrieves a value from the league's config."""
        return self.config.get(key, default)
    

    def set(self, key: str, value: Any) -> None:
        """Updates a single field in the league's config and writes it back to the file."""
        self.config[key] = value
        self._write_config()


    
    def set_file_path(self):
        self.filePath = "data/league_config.json"

