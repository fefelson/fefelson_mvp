from sqlalchemy import case
import pandas 

from ...database.models.games import Game
from ...database.models.game_lines import GameLine
from ...database.models.over_unders import OverUnder

from ...database.models.database import get_db_session


class GameLineStore:

    def __init__(self):

        pass

    def get_game_line_data(self, team_id: str):
        with get_db_session() as session:
            # Build the query
            query = (
                session.query(
                    Game.game_id,
                    Game.game_date.label('game_time'),
                    GameLine.opp_id,
                    case(
                        (GameLine.team_id == Game.home_team_id, 1),
                        else_=0
                    ).label('is_home'),
                    case(
                        (GameLine.team_id == Game.winner_id, 1),
                        else_=0
                    ).label('is_winner'),
                    GameLine.result,
                    GameLine.money_line.label('money'),
                    GameLine.spread,
                    GameLine.spread_outcome.label('is_cover'),
                )
                .select_from(GameLine)
                # Join with Game
                .join(
                    Game,
                    (GameLine.game_id == Game.game_id) &
                    ((GameLine.team_id == Game.away_team_id) | 
                    (GameLine.team_id == Game.home_team_id))
                )
                # Filters
                .filter(
                    GameLine.team_id == team_id
                )
                # Order by game_id descending
                .order_by(Game.game_id)
            )
            return pandas.DataFrame([row._asdict() for row in query.all()])
        

    def _moving_avg(self, data, n):
        # Calculate moving averages (6-game and 13-game)
        a, b = 6, 13
        valueList = []
        if len(data) >= n:
            valueList = [sum(data[(i-n):i])/n for i in range(n, len(data)+1)] # average of n number of games
        return valueList
        

    def get_win_loss_chart(self, gameLog):
        results = gameLog["result"]
        homeList = []
        awayList = []
        sixAvg = self._moving_avg(results, 6)
        thirteenAvg = self._moving_avg(results, 13)

        for i, row in gameLog.iterrows():
            if row.is_home:
                homeList.append((i, row.result)) # adds i for placement
            else:
                awayList.append((i, row.result))

        return homeList, awayList, sixAvg, thirteenAvg


    def get_pts_spread_chart(self, gameLog):
        ptsSpreads = [row.spread*-1 for _, row in gameLog.iterrows()]
        homeList = []
        awayList = []
        sixAvg = self._moving_avg(ptsSpreads, 6)
        thirteenAvg = self._moving_avg(ptsSpreads, 13)

        for i, row in gameLog.iterrows():
            if row.is_home:
                homeList.append((i, row.spread*-1)) # adds i for placement
            else:
                awayList.append((i, row.spread*-1)) # multiplies by -1 for clarity vs/ wins chart

        return homeList, awayList, sixAvg, thirteenAvg
    

    def get_ats_chart(self, gameLog):
        covers =[row.result+row.spread for _, row in gameLog.iterrows()]
        winList = []
        lossList = []
        sixAvg = self._moving_avg(covers, 6)
        thirteenAvg = self._moving_avg(covers, 13)

        for i, row in gameLog.iterrows():
            if row.is_winner:
                winList.append((i, (row.result+row.spread)+row.is_cover)) # adds i for placement
            else:
                lossList.append((i, (row.result+row.spread)+row.is_cover)) # adds is_cover to add depth to 0.5 case without affectig push case

        return winList, lossList, sixAvg, thirteenAvg
        

        

