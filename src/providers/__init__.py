from typing import Optional

from .yahoo.yahoo_downloader import YahooDownloadAgent
from .yahoo.normalizers.yahoo_basketball_normalizer import YahooBasketballNormalizer


default_provider = "yahoo"


def get_normal_agent(leagueId: str, provider: Optional[str]=None) -> "INormalAgent":
    if not provider:
        provider = default_provider

    return {"yahoo": {"NBA": YahooBasketballNormalizer,
                      "NCAAB": YahooBasketballNormalizer},
            }[provider][leagueId]



def get_download_agent(leagueId: str, provider: Optional[str]=None) -> "IDownloadAgent":
    if not provider:
        provider = default_provider
    
    return {"yahoo": {"NBA": YahooDownloadAgent,
                      "NCAAB": YahooDownloadAgent,
                      "NFL": YahooDownloadAgent,
                      "NCAAF": YahooDownloadAgent,
                      "MLB": YahooDownloadAgent}
            }[provider][leagueId]