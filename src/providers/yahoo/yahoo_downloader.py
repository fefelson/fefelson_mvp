from typing import Any, Dict
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import json

from ...agents import IDownloadAgent
from ...utils.logging_manager import get_logger


######################################################################
######################################################################

logger = get_logger()

class YahooDownloadAgent(IDownloadAgent):

    BASE_URL = "https://sports.yahoo.com"


    @staticmethod   
    def _fetch_url(url: str, sleepTime: int = 10, attempts: int = 3) -> Dict[str, Any]:
        """
        Recursive function to download yahoo url and isolate json
        Or write to errorFile
        """
        try:
            html = urlopen(url)
            for line in [x.decode("utf-8") for x in html.readlines()]:
                if "root.App.main" in line:
                    item = json.loads(";".join(line.split("root.App.main = ")[1].split(";")[:-1]))
                    item = item["context"]["dispatcher"]["stores"]
        
        except (URLError, HTTPError, ValueError) as e:
            logger.error(e)
            YahooDownloadAgent._fetch_url(url, sleepTime, attempts)
        return item
    

    @staticmethod
    def _form_scoreboard_url(leagueId: str, gameDate: str) -> str:
        slugId = {"NBA": "nba", "NCAAB": "college-basketball"}[leagueId]
        schedUrl = YahooDownloadAgent.BASE_URL+"/{0[slugId]}/scoreboard/?confId=all&schedState={0[schedState]}&dateRange={0[dateRange]}".format({"slugId":slugId, "schedState":"", "dateRange":gameDate})        
        return schedUrl
    

    @staticmethod
    def _form_boxscore_url(url: str) -> str:
        return YahooDownloadAgent.BASE_URL+url


        

       

    