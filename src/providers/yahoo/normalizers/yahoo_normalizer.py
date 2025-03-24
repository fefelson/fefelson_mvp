from datetime import datetime
from typing import List, Any, Dict
import pytz

from ....agents.normalize_agent import INormalAgent
from ....database.models.games import Game
from ....database.models.game_lines import GameLine
from ....database.models.over_unders import OverUnder
from ....database.models.periods import Period
from ....database.models.players import Player
from ....database.models.stadiums import Stadium
from ....database.models.teams import Team 
from ....models.model_classes import ScoreboardData, BoxscoreData, MatchupData
from ....utils.logging_manager import get_logger


est = pytz.timezone('America/New_York')


class YahooNormalizer(INormalAgent):
    """ normalizer for Yahoo data."""

    def __init__(self, leagueId: str, sportId: str):

        self.leagueId = leagueId
        self.sportId = sportId
        self.logger = get_logger()


    def _set_game_info(self, game: Dict[str, Any]) -> "Game":
            
        winnerId = game.get("winning_team_id")
        if winnerId:
            loserId = game["home_team_id"] if winnerId == game["away_team_id"] else game["away_team_id"]
            winnerId, loserId = winnerId, loserId
        else:
            loserId = None

        newGame = Game(
            game_id = game["gameid"],
            league_id = self.leagueId,
            home_team_id = game["home_team_id"],
            away_team_id = game["away_team_id"],
            winner_id = winnerId,
            loser_id = loserId,
            stadium_id = game.get("stadium_id", None),
            is_neutral_site = bool(game.get("tournament", 0)),
            game_date = str(datetime.strptime(game["start_time"], "%a, %d %b %Y %H:%M:%S %z").astimezone(est)), 
            season = game["season"],
            week = game.get("week", None),
            game_type = game["season_phase_id"].split(".")[-1],
            game_result = game["outcome_type"].split(".")[-1]
        )
        return newGame

        
    def _set_game_lines(self, data: Dict[str, Any]) -> List[GameLine]:
        gameId = data["gameid"]
        teamIds = {"away": data["away_team_id"], "home": data["home_team_id"]}
        oppIds = {"away": teamIds["home"], "home": teamIds["away"]}
        
        gameLines = []
        odds = list(data["odds"].values())[-1]

        awayPts = int(data["total_away_points"])
        homePts = int(data["total_home_points"])

        for a_h, teamPts, oppPts in [("away", awayPts, homePts), ("home", homePts, awayPts)]:
            try:
                result = teamPts - oppPts
                spread_key = f"{a_h}_spread"
                spreadOutcome = (result > float(odds[spread_key])) - (result < float(odds[spread_key]))
                moneyOutcome = teamPts > oppPts  # Boolean, automatically 1 (win) or 0 (loss)
                
                newGameLine = GameLine(
                    team_id=teamIds[a_h],
                    opp_id=oppIds[a_h],
                    game_id=gameId,
                    spread=odds[spread_key],
                    spread_line= -110 if odds[f"{a_h}_line"] == '' else odds[f"{a_h}_line"] ,
                    money_line=None if odds[f"{a_h}_ml"] == '' else odds[f"{a_h}_ml"], 
                    result=result,
                    spread_outcome=spreadOutcome,
                    money_outcome=None if odds[f"{a_h}_ml"] == '' else moneyOutcome
                )
                gameLines.append(newGameLine)
            except ValueError:
                pass
        return gameLines
    

    def _set_misc(self, webData: dict) -> Any:
        raise NotImplementedError


    def _set_over_under(self, data: Dict[str, Any]) -> OverUnder:
        gameId = data["gameid"]
        odds = list(data["odds"].values())[-1]
        total = int(data["total_away_points"]) + int(data["total_home_points"])

        try:
            overUnder = OverUnder(
                game_id=gameId,
                over_under=odds["total"],
                over_line=-110 if odds["over_line"] == '' else odds["over_line"],
                under_line=-110 if odds["under_line"] == '' else odds["under_line"],
                total=total,
                ou_outcome=(float(total) > float(odds["total"])) - (float(total) < float(odds["total"]))
                )
        except ValueError:
            pass
        return overUnder


    def _set_period_data(self, data: Dict[str, Any]) -> List[Period]:
        gameId = data["gameid"]
        teamIds = {"away": data["away_team_id"], "home": data["home_team_id"]}
        oppIds = {"away": teamIds["home"], "home": teamIds["away"]}

        periods = []
        for p in data["game_periods"]:
            periodId = p["period_id"]
            for a_h in ("away", "home"):
                newPeriod = Period(
                    game_id=gameId,
                    team_id=teamIds[a_h],
                    opp_id=oppIds[a_h],
                    period=periodId,
                    pts=p["{}_points".format(a_h)]
                )
                periods.append(newPeriod)
        return periods


    def _set_players(self, data: Dict[str, Any]) -> List[Player]:
        posTypes = dict([(key, value["abbr"]) for key, value in data["positions"].items()])
        
        players = []
        for key, value in data["players"].items():
            try:
                position = posTypes.get(value.get("primary_position_id", {}), None)
            except TypeError:
                position = None
            
            newPlayer = Player(
                player_id=value["player_id"],
                sport_id=self.sportId,
                first_name=value["first_name"],
                last_name=value["last_name"],
                position=position,
                current_team_id=value["team_id"],
                uniform_number=value.get("uniform_number", -1)
            )
            players.append(newPlayer)
        return players


    def _set_player_stats_list(self, data: dict) -> List:
        
        teamIds = {"away": data["away_team_id"], "home": data["home_team_id"]}
        oppIds = {"away": teamIds["home"], "home": teamIds["away"]}

        playerList = []
        for a_h in ("away", "home"):
            try:
                for playerId in data["lineups"]["{}_lineup_order".format(a_h)]["all"]:
                    playerList.append((playerId, teamIds[a_h], oppIds[a_h]))
            except (KeyError, TypeError) as e:
                self.logger.warning("No player List for "+data["gameid"])
        return playerList


    def _set_stadium(self, data: Dict[str, Any]) -> Stadium:
        return Stadium(
            stadium_id=data["stadium_id"],
            name=data.get("stadium", None)
            )


    def _set_teams(self, data: dict) -> List["Team"]:
        teams = []
        for team in [value for key, value in data.items() if self.leagueId.lower() in key]:
            newTeam = Team(
                team_id=team["team_id"],
                league_id=self.leagueId,
                first_name=team["first_name"],
                last_name=team["last_name"],
                abbreviation=team.get("abbr", "N/A"),
                conference=team.get("conference_abbr", None),
                division=team.get("division", None),
                primary_color=team.get("colorPrimary", None),
                secondary_color=team.get("colorSecondary", None)
                )
            teams.append(newTeam)
        return teams  
    

    def _set_lineup_data(self, data: dict) -> List["LineupData"]:
        raise NotImplementedError


    
    def _set_scoreboard_matchup(self, game: Dict[str, Any]) -> "MatchupData":
       
        try:
           odds = []
           for o in game["odds"].values():
               o["timestamp"] = str(datetime.now().astimezone(est))
               odds.append(o)
        except KeyError:
            odds = []

        try:
            url= game["navigation_links"]["boxscore"]["url"]
        except TypeError:
            url = None

        return MatchupData(
            provider="yahoo",
            gameId = game["gameid"],
            leagueId = self.leagueId,
            homeId = game["home_team_id"],
            awayId = game["away_team_id"],
            url = url,
            gameTime = str(datetime.strptime(game["start_time"], "%a, %d %b %Y %H:%M:%S %z").astimezone(est)), 
            season = game["season"],
            week = game.get("week", None),
            statusType=game["status_type"],
            gameType=game["game_type"].split(".")[-1] if game["game_type"] else None,
            odds=odds,
        )


    def _set_team_stats(self, data: dict) -> List["TeamStatData"]:
        raise NotImplementedError


    def _set_player_stats(self, data: dict) -> List["PlayerStatData"]:
        raise NotImplementedError

    
    def normalize_boxscore(self, webData: dict) -> "BoxscoreData":
        self.logger.debug("Normalize boxscore")

        gameId = webData["PageStore"]["pageData"]["entityId"]
        gameData = webData["GamesStore"]["games"][gameId]
        gameInfo = self._set_game_info(gameData)

        try:
            gameLines = self._set_game_lines(gameData)
            overUnder = self._set_over_under(gameData)
        except (KeyError, UnboundLocalError) as e:
            self.logger.warning(f"No Odds Data for {gameId}")
            gameLines, overUnder = None, None

        boxscore = BoxscoreData(
            game=gameInfo,
            teamStats=self._set_team_stats(webData),
            playerStats=self._set_player_stats(webData),
            periods=self._set_period_data(gameData),
            gameLines=gameLines,
            overUnders=overUnder,
            lineups=None,
            teams=[team for team in self._set_teams(webData["TeamsStore"]["teams"])
                   if team.team_id in (gameInfo.home_team_id, gameInfo.away_team_id)],
            players=self._set_players(webData["PlayersStore"]),
            stadium=self._set_stadium(gameData),
            misc=self._set_misc(webData)
        )
        return boxscore


    def normalize_matchup(self, webData: dict) -> "MatchupData":

        gameId = webData["PageStore"]["pageData"]["entityId"]
        gameData = webData["GamesStore"]["games"][gameId]

        try:
           odds = []
           for o in gameData["odds"].values():
               o["timestamp"] = str(datetime.now().astimezone(est))
               odds.append(o)
        except KeyError:
            odds = []

       
        return MatchupData(
            provider="yahoo",
            gameId = gameData["gameid"],
            leagueId = self.leagueId,
            homeId = gameData["home_team_id"],
            awayId = gameData["away_team_id"],
            url = gameData["navigation_links"]["boxscore"]["url"],
            gameTime = str(datetime.strptime(gameData["start_time"], "%a, %d %b %Y %H:%M:%S %z").astimezone(est)), 
            season = gameData["season"],
            week = gameData.get("week", None),
            statusType=gameData["status_type"],
            gameType=gameData["game_type"].split(".")[-1] if gameData["game_type"] else None,
            odds=odds,
            lineups=gameData.get("lineups", None),
            players=gameData["playersByTeam"] if gameData.get("playersByTeam") else None,
            teams = [webData["TeamsStore"]["teams"][teamId] for teamId in [gameData["{}_team_id".format(a_h)] for a_h in ("away", "home")]],
            injuries=[player["injury"] for player in webData["PlayersStore"]["players"].values() if player.get("injury", None)],
            stadiumId = gameData.get("stadium_id", None),
            isNuetral = bool(gameData.get("tournament", 0)),
        )
    

    def normalize_player(self, webData: dict) -> "Player":
        raise NotImplementedError


    def normalize_scoreboard(self, webData: dict) -> "ScoreboardData":

        leagueId = webData["PageStore"]["pageData"]["entityData"]["leagueName"]
        games = [self._set_scoreboard_matchup(game) for gameId, game in webData["GamesStore"]["games"].items() if gameId.split(".")[0] == leagueId.lower()]  
                    
        return ScoreboardData(provider="yahoo", 
                              league_id=leagueId,
                              games=games,
                              )          
    

    def normalize_team(self, raw_data: dict) -> "Team":
        raise NotImplementedError
        