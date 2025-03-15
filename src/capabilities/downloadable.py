from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

#################################################################
#################################################################


class Downloadable(ABC):
    """Enables downloading data from a URL"""
    
    def __init__(self, downloadAgent: Optional["IDownloadAgent"]=None):
        self.downloadAgent = downloadAgent  # Fetches raw data
        self.url = Optional[str] = None 


    def _set_download_agent(self, downloadAgent: "IDownloadAgent"):
        self.downloadAgent = downloadAgent


    def _set_url(self, url: str) -> None:
        if not isinstance(url, str):
            raise TypeError("url must be a string")
        self.url = url


    def download(self) -> Dict[str, Any]:
        """Fetches data from the URL and converts it into dict."""
        print(f"Downloadable.download called for {self.url}")
        return self.downloadAgent._fetch_url(self.url)
                       
    
    @abstractmethod
    def set_url(self, url: str=None) -> None:
        """Sets the URL to download from, optionally using provided item."""
        self._set_url(url)

    

   