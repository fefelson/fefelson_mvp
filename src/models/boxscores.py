from datetime import datetime
import os

from ..agents.file_agent import get_file_agent
from ..agents.database_agent import SQLAlchemyDatabaseAgent
from ..capabilities import Fileable, Normalizable, Processable, Downloadable, Databaseable
from ..models.model_classes import BoxscoreData, MatchupData
from ..providers import get_download_agent, get_normal_agent
from ..utils.logging_manager import get_logger

######################################################################
######################################################################



class Boxscore(Databaseable, Downloadable, Fileable, Normalizable, Processable):

    _fileType = "pickle"
    _fileAgent = get_file_agent(_fileType)
    _dbAgent = SQLAlchemyDatabaseAgent


    def __init__(self, leagueId: str):
        super().__init__()

        self.leagueId = leagueId
        self._set_file_Agent(self._fileAgent)
        self._set_dbAgent(self._dbAgent)

        self.logger = get_logger()


    def load_from_db(self) -> "Boxscore":
        """Loads data from the database into dataclass object using and returns it."""
        print(f"Databaseable.load_from_db called ")
        raise NotImplementedError


    def normalize(self, webData: dict) -> "BoxscoreData":
        return self.normalAgent.normalize_boxscore(webData)


    def process(self, game: "MatchupData") -> "BoxscoreData":
        self.logger.debug("processing Boxscore")
        
        self.set_file_path(game)
        if self.file_exists():
            boxscore = self.read_file()
        else:
            self._set_download_agent(get_download_agent(self.leagueId, game.provider))
            if game.url:
                self.set_url(game)
                webData = self.download()

                normalAgent = get_normal_agent(self.leagueId, game.provider)
                self._set_normal_agent(normalAgent(self.leagueId))
                boxscore = self.normalize(webData)
                self.write_file(boxscore)
        self.save_to_db(boxscore)        
    

    def save_to_db(self, boxscore: "Boxscore"):
        """Saves self.data to the database."""
        try:
            self.dbAgent.insert_boxscore(boxscore)
        except Exception as e:
            # Catch unexpected errors
            self.logger.error(f"Failed to save boxscore to db: Unexpected error - {type(e).__name__}: {str(e)}")
            # raise  # Optional: re-raise for debugging
        else:
            # Runs if no exception occurs
            self.logger.info("Boxscore saved to db successfully")
        # Optional: Add a finally block if you need cleanup
        

    def set_url(self, game: "MatchupData"):
        self.url = self.downloadAgent._form_boxscore_url(game.url)


    def set_file_path(self, game: "MatchupData"):
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

        self.filePath = os.path.join(module_dir, '..', '..', 'data', "boxscores", filePath.format(params))


    
