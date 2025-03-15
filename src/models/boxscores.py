from typing import List, Optional

from ..capabilities import Downloadable, Fileable, Normalizable, Processable
from ..providers import get_download_agent, get_normal_agent

######################################################################
######################################################################



class Boxscore(Downloadable, Fileable, Normalizable, Processable):

    def __init__(self, leagueId: str):
        self.leagueId = leagueId


    def normalize(self, webData: dict) -> "BoxscoreData":
        return self.normalAgent.normalize_boxscore(webData)


    def process(self, game: "GameData"):
        self.setUrl(game)
        self.setFilePath(game)
        if self.file_exists():
            boxscore = self.read_file()
        else:
            downloadAgent = get_download_agent(game.provider, self.leagueId)
            normalAgent = get_normal_agent(game.provider, self.leagueId)
            self._set_download_agent(downloadAgent)
            self._set_normal_agent(normalAgent)
            webData = self.download()
            boxscore = self.normalize(webData)
            self.write_file(boxscore)
        return boxscore


    def set_url(self, game: "GameData"):
        self.url = self.downloadAgent._form_boxscore_url(game.url)


    def set_file_path(self, game: "GameData"):
        pass
