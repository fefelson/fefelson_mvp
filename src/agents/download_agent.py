from abc import ABC, abstractmethod


class IDownloadAgent(ABC):
    
    def fetch_url(self, url: str):
        pass

