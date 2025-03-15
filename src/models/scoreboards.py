from typing import List, Optional

from ..capabilities import Downloadable, Normalizable, Processable
from ..providers import get_download_agent, get_normal_agent

####################################################################
####################################################################


class Scoreboard(Downloadable, Normalizable, Processable):

    def __init__(self, leagueId: str):
        self.leagueId = leagueId


    def normalize(self, webData: dict) -> "ScoreboardData":
        return self.normalAgent.normalize_scoreboard(webData)


    def process(self, gameDate: str):
        self._set_download_agent(get_download_agent(self.leagueId))
        self._set_normal_agent(get_normal_agent(self.leagueId))
        self.set_url(gameDate)
        webData = self.download()
        scoreboard = self.normalize(webData)
        return scoreboard


    def set_url(self, gameDate: str):
        self.url = self.downloadAgent._form_scoreboard_url(self.leagueId, gameDate)







    
