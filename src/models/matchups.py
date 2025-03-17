from datetime import datetime, timedelta, timezone

from ..agents.file_agent import get_file_agent
from ..capabilities import Fileable, Normalizable, Processable, Downloadable
from ..providers import get_download_agent, get_normal_agent
from ..utils.logging_manager import get_logger

######################################################################
######################################################################



class Matchup(Downloadable, Fileable, Normalizable, Processable):

    _fileType = "pickle"
    _fileAgent = get_file_agent(_fileType)


    def __init__(self, leagueId: str):
        super().__init__()

        self.leagueId = leagueId
        self.logger = get_logger()
        self._set_file_Agent(self._fileAgent)


    def needs_update(self, matchup):
        return (matchup.gameTime - datetime.now(timezone.utc) < timedelta(hours=3)) and not matchup.lineups


    def normalize(self, webData: dict) -> "MatchupData":
        return self.normalAgent.normalize_matchup(webData)


    def process(self, game: "GameData") -> "MatchupData":
        self.logger.info("process Matchup")
        self.set_file_path(game)
        if self.file_exists():
            matchup = self.read_file()
            if self.needs_update(matchup):
                self.update(matchup)
            else:
                [matchup.odds.append(odds) for odds in game.odds]
        else:
            matchup = self.update(game)
        return matchup
    

    def set_file_path(self, game: "GameData"):
        basePath = "data/matchups/{0[leagueId]}"
        params = {"leagueId": game.leagueId, 
                  "fileName": game.gameId.split(".")[-1],
                  "ext": self.fileAgent.get_ext()}
        filePath = basePath+"/{0[fileName]}.{0[ext]}"

        self.filePath = filePath.format(params)


    def set_url(self, game: "GameData"):
        self.url = self.downloadAgent._form_boxscore_url(game.url)
    

    def update(self, matchup):
        self._set_download_agent(get_download_agent(self.leagueId, matchup.provider))
        if matchup.url:
            self.set_url(matchup)
            
            webData = self.download()

            normalAgent = get_normal_agent(self.leagueId, matchup.provider)
            self._set_normal_agent(normalAgent(self.leagueId))
            tempMatchup = self.normalize(webData)

            if tempMatchup.injuries:
                matchup.injuries = tempMatchup.injuries

            if tempMatchup.lineups:
                matchup.lineups = tempMatchup.lineups 

            [matchup.odds.append(odds) for odds in tempMatchup.odds]
        self.write_file(matchup)
        return matchup


    


    

