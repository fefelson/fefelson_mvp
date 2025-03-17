from ...config.config_manager import LeagueConfig
from ...models.leagues import League
from ...models.schedules import DailySchedule


####################################################################
####################################################################


class NBA(League):

    _leagueId = "NBA"
    _leagueConfig = LeagueConfig(_leagueId)    
    _schedule = DailySchedule



####################################################################
####################################################################


class NCAAB(League):

    _leagueId = "NCAAB"
    _leagueConfig = LeagueConfig(_leagueId)    
    _schedule = DailySchedule

    