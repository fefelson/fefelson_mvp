from ...config.config_manager import LeagueConfig
from ...models.analytics import MLBAnalytics
from ...models.leagues import League
from ...models.schedules import DailySchedule


####################################################################
####################################################################


class MLB(League):

    _leagueId = "MLB"
    _leagueConfig = LeagueConfig(_leagueId)
    _analytics = MLBAnalytics()    
    _schedule = DailySchedule


