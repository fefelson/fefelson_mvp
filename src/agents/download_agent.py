from abc import ABC, abstractmethod


class IDownloadAgent(ABC):
    
    @abstractmethod
    def fetch_url(url: str):
        pass

    @abstractmethod
    def _form_scoreboard_url(url: str):
        pass 

    @abstractmethod
    def _form_boxscore_url(url: str):
        pass

