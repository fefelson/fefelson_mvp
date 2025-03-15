from typing import Any, List, Optional

from src.capabilities import Databaseable, Processable, Updateable
from src.config import get_logger, LeagueConfig
from src.models import Boxscore, Scoreboard, DailySchedule

################################################################################
################################################################################


class League(Databaseable, Processable, Updateable):

    _leagueId = None
    _schedule = None


    def __init__(self, dbAgent: Optional["IDatabaseAgent"] = None):
        Databaseable.__init__(self, dbAgent)

        self.config = LeagueConfig(self._leagueId)
        self.schedule = self._schedule(self._leagueId)

        self.scoreboard = Scoreboard(self._leagueId)
        self.boxScore = Boxscore(self._leagueId)
        # self.matchup = self._matchup(self._leagueId)
        # self.player = self._player(self._leagueId)
        # self.team = self._team(self._leagueId)

        self.logger = get_logger()


    def _collect_boxscores(self, scoreboard: "ScoreboardData") -> List["BoxscoreData"]:
        boxscoreList = []
        for game in [g for g in scoreboard.games if g.final == True and g.game_type != "preseason"]:
            boxscoreList.append(self.boxscore.process(game))
        return boxscoreList
                

    def _collect_matchups(self, scoreboard: "ScoreboardData") -> List["MatchupData"]:
        matchupList = []
        for game in scoreboard.games:
              matchupList.append(self.matchup.process(game))
        return matchupList


    def load_from_db(self) -> Optional[List[Any]]:
        """Loads data from the database into dataclass object using and returns it."""
        print(f"Databaseable.load_from_db called ")
        raise NotImplementedError
    

    def needs_update(self):
        return self.schedule.is_active()


    def process(self):
        if self.needs_update():
            self.update()


    def save_to_db(self, identifier: str, dataList: List[Any]):
        """Saves self.data to the database."""
        print(f"Databaseable.save_to_db called for identifier: {identifier}")
        if identifier == "boxscores":
            try:
                self.dbAgent.process_boxscores(dataList)
            except:
                self.logger.error("saving boxscores to db failed")
                raise
            self.logger.info("boxscores saved to db successfully")


    def update(self):
        self.logger.info(f"Updating {self._leagueId}")
        lastUpdate = self.config.get("lastUpdate")
        from datetime import datetime
        lastUpdate = str(datetime.today().date())
        
        for gameDate in self.schedule.process(lastUpdate, nGD=2):
            scoreboard = self.scoreboard.process(gameDate)
            boxscoreList = self._collect_boxscores(scoreboard)
            self.save_to_db("boxscores", boxscoreList) 
            self.config.set("lastUpdate", gameDate)
            self.config.write_config()
            self.logger.info(f"{self._leagueId} current up until {gameDate}")
        
        for gameDate in self.schedule.process():
             matchupList = self._collect_matchups(gameDate)
             self.logger.info(f"collected {self._leagueId} Matchups for {gameDate}")

        self.logger.info(f"{self._leagueId} is up to date")



####################################################################
####################################################################


class NBA(League):

    _leagueId = "NBA"
    _schedule = DailySchedule



