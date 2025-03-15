from ...models import (DailySchedule, League)


####################################################################
####################################################################



class NBA(League):

    _leagueId = "NBA"
    _schedule = DailySchedule

    