from typing import List

from ..capabilities import Downloadable, Normalizable, Processable
from ..utils.logging_manager import get_logger


####################################################################
####################################################################


class Scoreboard(Downloadable, Normalizable, Processable):

    def __init__(self, leagueId: str):
        self.leagueId = leagueId
        self.logger = get_logger()


    def normalize(self, webData: dict) -> "ScoreboardData":
        return self.normalAgent.normalize_scoreboard(webData)


    def process(self, gameDate: str) -> List["GameData"]:
        self.logger.info(f"{self.leagueId} Scoreboard processing {gameDate}")
        from ..providers import get_download_agent, get_normal_agent # factory methods
        self._set_download_agent(get_download_agent(self.leagueId)) # Using factory method
        normalAgent = get_normal_agent(self.leagueId)
        self._set_normal_agent(normalAgent(self.leagueId)) # Using factory method
        
        self.set_url(gameDate)
        scoreboard = self.normalize(self.download())
        return scoreboard.games


    def set_url(self, gameDate: str):
        self.url = self.downloadAgent._form_scoreboard_url(self.leagueId, gameDate)

    
