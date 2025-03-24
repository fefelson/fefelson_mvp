from datetime import datetime, timedelta
from dataclasses import asdict
import os
import pytz

from .model_classes import MatchupData
from ..agents.file_agent import get_file_agent
from ..capabilities import Fileable, Normalizable, Processable, Downloadable
from ..providers import get_download_agent, get_normal_agent
from ..utils.logging_manager import get_logger

######################################################################
######################################################################


est = pytz.timezone('America/New_York')

class Matchup(Downloadable, Fileable, Normalizable, Processable):

    _fileType = "json"
    _fileAgent = get_file_agent(_fileType)


    def __init__(self, leagueId: str):
        super().__init__()

        self.leagueId = leagueId
        self.logger = get_logger()
        self._set_file_Agent(self._fileAgent)


    def needs_update(self, matchup):
        return (datetime.fromisoformat(matchup.gameTime)- datetime.now().astimezone(est) < timedelta(hours=3)) and not matchup.lineups


    def normalize(self, webData: dict) -> "MatchupData":
        return self.normalAgent.normalize_matchup(webData)


    def process(self, game: "GameData") -> "MatchupData":
        self.logger.info("process Matchup")
        
        self.set_file_path(game)
        if self.file_exists():
            matchup = MatchupData(**self.read_file())
            if self.needs_update(matchup):
                self.update(matchup)
            else:
                [matchup.odds.append(odds) for odds in game.odds]
        else:
            matchup = self.update(game)
        self.write_file(matchup)
        return matchup
    
 
    def set_file_path(self, game: "GameData"):
        module_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up two levels to ~/fefelson_mvp, then into data
        basePath = "{0[leagueId]}/{0[season]}"
        dailyPath = "/{0[month]}/{0[day]}/"
        weeklyPath ="/{0[week]}/"

        params = {"leagueId": game.leagueId, 
                  "season": game.season, 
                  "fileName": game.gameId.split(".")[-1],
                  "ext": self.fileAgent.get_ext()}
         
        if game.week:
            params["week"] = game.week
            filePath = basePath+weeklyPath
        else:
            params["month"], params["day"] = str(datetime.fromisoformat(game.gameTime).date()).split("-")[1:]
            filePath = basePath+dailyPath
        filePath += "/{0[fileName]}.{0[ext]}"

        self.filePath = os.path.join(module_dir, '..', '..', 'data', "matchups", filePath.format(params))



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

            if tempMatchup.players:
                matchup.players = tempMatchup.players

            if tempMatchup.teams:
                matchup.teams = tempMatchup.teams 

            if tempMatchup.injuries:
                matchup.injuries = tempMatchup.injuries

            if tempMatchup.lineups:
                matchup.lineups = tempMatchup.lineups 

            [matchup.odds.append(odds) for odds in tempMatchup.odds]
        return matchup
    

    def write_file(self, fileableObj) -> None:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.filePath), exist_ok=True)
        self.fileAgent.write(self.filePath, asdict(fileableObj))

    

    

