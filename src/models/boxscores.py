from ..agents.file_agent import get_file_agent
from ..agents.database_agent import SQLAlchemyDatabaseAgent
from ..capabilities import Fileable, Normalizable, Processable, Downloadable, Databaseable
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
        self.logger = get_logger()
        self._set_file_Agent(self._fileAgent)
        self._set_dbAgent(self._dbAgent)


    def load_from_db(self) -> "Boxscore":
        """Loads data from the database into dataclass object using and returns it."""
        print(f"Databaseable.load_from_db called ")
        raise NotImplementedError


    def normalize(self, webData: dict) -> "BoxscoreData":
        return self.normalAgent.normalize_boxscore(webData)


    def process(self, game: "GameData") -> "BoxscoreData":
        self.logger.info("process Boxscore")
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
        except:
            self.logger.error("saving boxscore to db failed")
            raise
        self.logger.info("boxscore saved to db successfully")


    def set_url(self, game: "GameData"):
        self.url = self.downloadAgent._form_boxscore_url(game.url)


    def set_file_path(self, game: "GameData"):
        basePath = "data/boxscores/{0[leagueId]}/{0[season]}"
        dailyPath = "/{0[month]}/{0[day]}"
        weeklyPath ="/{0[week]}/{0[fileName]}"

        params = {"leagueId": game.leagueId, 
                  "season": game.season, 
                  "fileName": game.gameId.split(".")[-1],
                  "ext": self.fileAgent.get_ext()}
         
        if game.week:
            params["week"] = game.week
            filePath = basePath+weeklyPath
        else:
            params["month"], params["day"] = str(game.game_date.date()).split("-")[1:]
            filePath = basePath+dailyPath
        filePath += "/{0[fileName]}.{0[ext]}"

        self.filePath = filePath.format(params)


    
