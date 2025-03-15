from datetime import datetime
from typing import List, Any, Dict, Tuple

from ....models.model_classes import GameData, ScoreboardData
from ....agents.normalize_agent import INormalAgent


class YahooNormalizer(INormalAgent):
    """ normalizer for Yahoo data."""


    def __init__(self):

        self.gameId = None
        self.teamIds = None
        self.oppIds = None
        self.statTypes = None
        self.posTypes = None


    # def _set_game_info(self, data: Dict[str, Any]) -> GameData:
    #     winnerId = data.get("winning_team_id")
    #     if winnerId:
    #         loserId = self.teamIds["home"] if winnerId == self.teamIds["away"] else self.teamIds["away"]
    #         winnerId, loserId = winnerId.split(".")[-1], loserId.split(".")[-1]
    #     else:
    #         loserId = None
        
    #     gameInfo = Games(
    #         game_id=data["gameid"].split(".")[-1],
    #         league_id=self.leagueId,
    #         home_team_id=data["home_team_id"].split(".")[-1],
    #         away_team_id=data["away_team_id"].split(".")[-1],
    #         winner_id=winnerId,
    #         loser_id=loserId,
    #         stadium_id=data.get("stadium_id", None),
    #         is_neutral_site=bool(data.get("tournament", 0)),
    #         game_date=datetime.strptime(data["start_time"], "%a, %d %b %Y %H:%M:%S %z"),
    #         season=data["season"],
    #         game_type=data["season_phase_id"].split(".")[-1],
    #         game_result=x["outcome_type"].split(".")[-1] if data.get("outcome_type", None) else data.get("outcome_type", None),
    #         url=data["navigation_links"]["boxscore"]["url"]
    #     )
    #     return gameInfo



    # def _set_game_lines(self, data: Dict[str, Any]) -> List[GameLines]:
    #     gameLines = []
    #     odds = list(data["gameData"]["odds"].values())[-1]

    #     awayPts = int(data["gameData"]["total_away_points"])
    #     homePts = int(data["gameData"]["total_home_points"])

    #     for a_h, teamPts, oppPts in [("away", awayPts, homePts), ("home", homePts, awayPts)]:
    #         try:
    #             result = teamPts - oppPts
    #             spread_key = f"{a_h}_spread"
    #             spreadOutcome = (result > float(odds[spread_key])) - (result < float(odds[spread_key]))
    #             moneyOutcome = teamPts > oppPts  # Boolean, automatically 1 (win) or 0 (loss)
                
    #             newGameLine = GameLines(
    #                 team_id=self.teamIds[a_h].split(".")[-1],
    #                 opp_id=self.oppIds[a_h].split(".")[-1],
    #                 game_id=self.gameId.split(".")[-1],
    #                 spread=odds[spread_key],
    #                 spread_line= -110 if odds[f"{a_h}_line"] == '' else odds[f"{a_h}_line"] ,
    #                 money_line=None if odds[f"{a_h}_ml"] == '' else odds[f"{a_h}_ml"], 
    #                 result=result,
    #                 spread_outcome=spreadOutcome,
    #                 money_outcome=None if odds[f"{a_h}_ml"] == '' else moneyOutcome
    #             )
    #             gameLines.append(newGameLine)
    #         except ValueError:
    #             #TODO: Log This
    #             pass
            
    #     return gameLines


    # def _set_over_under(self, data: Dict[str, Any]) -> OverUnders:
    #     odds = list(data["gameData"]["odds"].values())[-1]
    #     total = int(data["gameData"]["total_away_points"]) + int(data["gameData"]["total_home_points"])

    #     try:
    #         overUnder = OverUnders(
    #             game_id=self.gameId.split(".")[-1],
    #             over_under=odds["total"],
    #             over_line=-110 if odds["over_line"] == '' else odds["over_line"],
    #             under_line=-110 if odds["under_line"] == '' else odds["under_line"],
    #             total=total,
    #             ou_outcome=(float(total) > float(odds["total"])) - (float(total) < float(odds["total"]))
    #             )
    #     except ValueError:
    #         #TODO: Log This
    #         pass
    #     return overUnder


    # def _set_period_data(self, data: Dict[str, Any]) -> List[Periods]:
    #     periods = []
    #     for p in data["gameData"]["game_periods"]:
    #         periodId = p["period_id"]
    #         for a_h in ("away", "home"):
    #             newPeriod = Periods(
    #                 game_id=self.gameId.split(".")[-1],
    #                 team_id=self.teamIds[a_h].split(".")[-1],
    #                 opp_id=self.oppIds[a_h].split(".")[-1],
    #                 period=periodId,
    #                 pts=p["{}_points".format(a_h)]
    #             )
    #             periods.append(newPeriod)
    #     return periods


    # def _set_players(self, data: Dict[str, Any]) -> List[Players]:
    #     players = []
    #     for key, value in data["playerData"]["players"].items():
    #         try:
    #             position = self.posTypes.get(value.get("primary_position_id", {}), None)
    #         except TypeError:
    #             #TODO: logging
    #             position = None
            
    #         newPlayer = Players(
    #             player_id=value["player_id"].split(".")[-1],
    #             sport_id=self.sportId,
    #             first_name=value["first_name"],
    #             last_name=value["last_name"],
    #             position=position,
    #             current_team_id=value["team_id"].split(".")[-1],
    #             uniform_number=value.get("uniform_number", -1)
    #         )
    #         players.append(newPlayer)
    #     return players


    # def _set_player_stats_list(self, data: dict) -> List:
    #     playerList = []
    #     for a_h in ("away", "home"):
    #         try:
    #             for playerId in data["gameData"]["lineups"]["{}_lineup_order".format(a_h)]["all"]:
    #                 playerList.append((playerId, self.teamIds[a_h], self.oppIds[a_h]))
    #         except (KeyError, TypeError) as e:
    #             #TODO: Log This
    #             pass
    #     return playerList


    # def _set_stadium(self, data: Dict[str, Any]) -> Stadiums:
    #     return Stadiums(
    #         stadium_id=data["gameData"]["stadium_id"],
    #         name=data["gameData"].get("stadium", None)
    #         )


    def _set_teams(self, data: Dict[str, Any]) -> List[Teams]:
        teams = []
        for a_h in ("away", "home"):
            raw_stat_data = data["teamData"]["teams"][self.teamIds[a_h]]
            newTeam = Teams(
                team_id=self.teamIds[a_h].split(".")[-1],
                league_id=self.leagueId,
                first_name=raw_stat_data["first_name"],
                last_name=raw_stat_data["last_name"],
                abbreviation=raw_stat_data["abbr"],
                conference=raw_stat_data.get("conference", None),
                division=raw_stat_data.get("division", None),
                primary_color=raw_stat_data.get("colorPrimary", None),
                secondary_color=raw_stat_data.get("colorSecondary", None)
                )
            teams.append(newTeam)
        return teams  
    

    # def _standard_format(self, data: dict) -> None:
    #     self.gameId = data["gameData"]["gameid"]

    #     self.teamIds = {
    #         "away": data["gameData"]["away_team_id"], 
    #         "home": data["gameData"]["home_team_id"]
    #     }
    #     self.oppIds = {"away": self.teamIds["home"], "home": self.teamIds["away"]}
    #     self.statTypes = data["statsData"]["statTypes"]
    #     self.posTypes = dict([(key, value["abbr"]) for key, value in data["playerData"]["positions"].items()])


    # @abstractmethod
    # def _set_lineup_data(self, boxscore: Any, data: Dict[str, Any]) -> None:
    #     raise AssertionError


    # @abstractmethod
    # def _set_team_stats(self, boxscore: Any, data: Dict[str, Any]) -> None:
    #     raise AssertionError


    # @abstractmethod
    # def _set_player_stats(self, boxscore: Any, data: Dict[str, Any]) -> None:
    #     raise AssertionError


    # @abstractmethod
    # def _set_misc(self, data: Dict[str, Any]) -> List[Tuple]:
    #     raise AssertionError

    
    # def normalize_boxscore(self, data: Dict[str, Any]) -> Boxscore:
    #     self._standard_format(data)

    #     gameInfo = self._set_game_info(data["gameData"])
    #     teams = self._set_teams(data)
    #     players = self._set_players(data)

    #     try:
    #         gameLines= self._set_game_lines(data)
    #         overUnder= self._set_over_under(data)
    #     except KeyError as e:
    #         #TODO: logging
    #         pass

    #     boxscore = Boxscores(
    #         games= gameInfo,
    #         team_stats= self._set_team_stats(data),
    #         player_stats= self._set_player_stats(data),
    #         periods= self._set_period_data(data),
    #         game_lines= gameLines,
    #         over_unders= overUnder,
    #         lineups= self._set_lineup_data(data),
    #         teams= teams,
    #         players= players,
    #         stadiums= self._set_stadium(data)
    #     )
    #     for key, value in self._set_misc(data):
    #         boxscore[key] = value
    #     return boxscore


    # def normalize_matchup(self, raw_data: dict) -> Matchup:
    #     raise AssertionError
    

    # def normalize_player(self, raw_data: dict) -> Players:
    #     raise AssertionError


    def normalize_boxscore(self):
        raise NotImplementedError


    def normalize_scoreboard(self, raw_data: dict) -> List[GameData]:

        from pprint import pprint
    
        pprint(raw_data["TeamsStore"]["teams"])
        raise

        leagueId = raw_data["PageStore"]["pageData"]["entityData"]["leagueName"]
        gameDate = raw_data["ClientStore"]["currentRoute"]["query"]["dateRange"]

        games = []
        for game in [value for key, value in raw_data["GamesStore"]["games"].items() if key.split(".")[0] == leagueId.lower()]:
    
            winnerId = game.get("winning_team_id")
            if winnerId:
                loserId = game["home_team_id"] if winnerId == game["away_team_id"] else game["away_team_id"]
                winnerId, loserId = winnerId, loserId
            else:
                loserId = None

            newGame = GameData(
                provider="yahoo",
                game_id=game["gameid"],
                league_id=leagueId,
                home_team_id=game["home_team_id"],
                away_team_id=game["away_team_id"],
                winner_id=winnerId,
                loser_id=loserId,
                game_date=datetime.strptime(game["start_time"], "%a, %d %b %Y %H:%M:%S %z"), 
                season=game["season"],
                game_type=game["season_phase_id"],
                game_result=game.get("outcome_type",None),
                odds=game.get("odds", None),
                week=game.get("week", None),
                url=game["navigation_links"]["boxscore"]["url"],
                stadium_id=game.get("stadium_id", None),
                is_neutral_site=bool(game.get("tournament", 0))                
            )
            pprint(newGame)
            raise
            games.append(newGame)
            
        return ScoreboardData(provider="yahoo", 
                              leagueId=leagueId,
                              game_date=gameDate,
                              games=games)          
    

    # def normalize_team(self, raw_data: dict) -> Teams:
    #     raise AssertionError
        

            

        
        
        
       