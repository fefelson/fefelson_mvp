from ...config.config_manager import LeagueConfig
from ...models.analytics import NBAAnalytics, NCAABAnalytics
from ...models.leagues import League
from ...models.schedules import DailySchedule


####################################################################
####################################################################


class NBA(League):

    _leagueId = "NBA"
    _leagueConfig = LeagueConfig(_leagueId)
    _analytics = NBAAnalytics()    
    _schedule = DailySchedule



####################################################################
####################################################################


class NCAAB(League):

    _leagueId = "NCAAB"
    _leagueConfig = LeagueConfig(_leagueId)   
    _analytics = NCAABAnalytics()  
    _schedule = DailySchedule

    