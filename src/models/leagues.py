from .boxscores import Boxscore
from .matchups import Matchup
from .scoreboards import Scoreboard 
from ..capabilities import Processable, Updateable
from ..utils.logging_manager import get_logger


################################################################################
################################################################################


class League(Processable, Updateable):

    _leagueId = None
    _leagueConfig = None
    _schedule = None

    def __init__(self):
        super().__init__()
        
        self.boxscore = Boxscore(self._leagueId)
        self.matchup = Matchup(self._leagueId)
        # self.player = Player(self._leagueId)
        self.scoreboard = Scoreboard(self._leagueId)
        # self.team = Team(self._leagueId)

        self.logger = get_logger()


    def needs_update(self):
        return self._schedule.is_active(self._leagueConfig)


    def process(self):
        if self.needs_update():
            self.update()


    def update(self):
        self.logger.info(f"Updating {self._leagueId}")
        for gameDate in self._schedule.process(self._leagueConfig):
            self.logger.info(f"Next gameDate {gameDate}")
            for game in self.scoreboard.process(gameDate):
                if game.statusType == "final" and game.gameType != "Preseason":
                   self.boxscore.process(game)
            self._leagueConfig.set("last_update", gameDate)
            self._leagueConfig._write_config()
            self.logger.warning(f"{self._leagueId} current up until {gameDate}")


        for gameDate in self._schedule.process(self._leagueConfig, nGD=2):
            self.logger.info(f"Next gameDate {gameDate}")
            for game in self.scoreboard.process(gameDate):
                if game.statusType ==  "pregame" and game.gameType != "Preseason":
                    self.matchup.process(game)
            self.logger.warning(f"{self._leagueId} matchups processed for {gameDate}")
    
        self.logger.info(f"{self._leagueId} is up to date")
