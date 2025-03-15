import math
from typing import Any, Dict

from .yahoo_normalizer import YahooNormalizer

from ....core.interfaces.id_mapper import IIdMapper
from ....sports.basketball.normalizers import BasketballNormalizer


class YahooNBANormalizer(BasketballNormalizer, YahooNormalizer):
    """Concrete normalizer for Yahoo NBA Basketball data."""


    def __init__(self, idMapper: IIdMapper):
        super().__init__(idMapper)

        self.leagueId = "NBA"
        self.sportId = 0


    def _set_lineup_data(self, boxscore: Any, data: Dict[str, Any]) -> None:
        pass


    def _set_team_stats(self, boxscore: Any, data: Dict[str, Any]) -> None:
        teamStats = []
        for a_h in ("away", "home"):
            raw_stat_data = data["statsData"]["teamStatsByGameId"][self.gameId][self.teamIds[a_h]]["nba.stat_variation.2"]

            newTeamStats = self._TeamStats(
                game_id = self.gameId.split(".")[-1],
                team_id = self.teamIds[a_h].split(".")[-1],
                opp_id = self.oppIds[a_h].split(".")[-1],
                minutes=48+(len(data["gameData"]["game_periods"])-4)*5,
                fga = raw_stat_data["nba.stat_type.104"],
                fgm = raw_stat_data["nba.stat_type.105"],
                fta = raw_stat_data["nba.stat_type.107"],
                ftm = raw_stat_data["nba.stat_type.108"],
                tpa = raw_stat_data["nba.stat_type.110"],
                tpm = raw_stat_data["nba.stat_type.111"],
                pts = raw_stat_data["nba.stat_type.113"],
                oreb = raw_stat_data["nba.stat_type.114"],
                dreb = raw_stat_data["nba.stat_type.115"],
                ast = raw_stat_data["nba.stat_type.117"],
                stl = raw_stat_data["nba.stat_type.118"],
                blk = raw_stat_data["nba.stat_type.119"],
                turnovers = raw_stat_data["nba.stat_type.120"],
                fouls = raw_stat_data["nba.stat_type.122"],
##                fb_pts = sum(int(x["points"]) * int(x["shot_made"]) * int(x["fastbreak"]) for x in data["gameData"]["play_by_play"].values() if x["class_type"] == "SHOT" and x["team"] == self.teamIds[a_h].split(".")[-1]),
                pts_in_pt = sum(int(x["points"]) * int(x["shot_made"]) for x in data["gameData"]["play_by_play"].values() if x["class_type"] == "SHOT" and x["team"] == self.teamIds[a_h].split(".")[-1] and float(x["sideline_offset_percentage"]) <= 0.15 and float(x["baseline_offset_percentage"]) <= 0.4)
                )
            teamStats.append(newTeamStats)
        boxscore.team_stats=teamStats  


    def _set_player_stats(self, boxscore: Any, data: Dict[str, Any]) -> None:
        playerStats = []
        starters = [posRecord["player_id"] for a_h in ("away", "home") for posRecord in data["gameData"]["lineups"]["{}_lineup".format(a_h)]["all"].values() if int(posRecord["starter"]) == 1]
        
        for playerId, teamId, oppId in self._set_player_stats_list(data):

            try:
                raw_player_data = data["statsData"]["playerStats"][playerId]["nba.stat_variation.2"]
            except KeyError:
                # Did Not Play
                raw_player_data = None
                
            if raw_player_data:
                newPlayerStats = self._PlayerStats(
                    player_id=playerId.split(".")[-1],
                    game_id=self.gameId.split(".")[-1],
                    team_id=teamId.split(".")[-1],
                    opp_id=oppId.split(".")[-1],
                    starter=(playerId in starters),
                    mins = f"{(t := raw_player_data.get('nba.stat_type.3', '0:0').split(':'))[0]}.{int((int(t[1]) / 60) * 100 + 0.5) if len(t) > 1 else 0}",
                    fgm=raw_player_data["nba.stat_type.28"].split("-")[0],
                    fga=raw_player_data["nba.stat_type.28"].split("-")[1],
                    ftm=raw_player_data["nba.stat_type.29"].split("-")[0],
                    fta=raw_player_data["nba.stat_type.29"].split("-")[1],
                    tpm=raw_player_data["nba.stat_type.30"].split("-")[0],
                    tpa=raw_player_data["nba.stat_type.30"].split("-")[1],
                    pts=raw_player_data["nba.stat_type.13"],
                    oreb=raw_player_data["nba.stat_type.14"],
                    dreb=raw_player_data["nba.stat_type.15"],
                    ast=raw_player_data["nba.stat_type.17"],
                    stl=raw_player_data["nba.stat_type.18"],
                    blk=raw_player_data["nba.stat_type.19"],
                    turnovers=raw_player_data["nba.stat_type.20"],
                    fouls=raw_player_data["nba.stat_type.22"],
                    plus_minus=raw_player_data["nba.stat_type.32"]
                )
                playerStats.append(newPlayerStats)
        boxscore.player_stats=playerStats


    def _set_player_shots(self, boxscore: dict, data: dict) -> None:
        """Convert a shot record dictionary to BasketballPlayerShots dataclass if class_type is SHOT."""

        playerShots = []
        for shot in [shot for shot in data["gameData"]["play_by_play"].values() if shot["class_type"] == "SHOT" and int(shot["type"]) not in range(10,25)]:

            # Calculate distance using base_pct and side_pct (court: 50ft x 94ft)
            base_pct = float(shot['baseline_offset_percentage'])
            side_pct = float(shot['sideline_offset_percentage'])
            base_pct_adjusted = base_pct * ((-1) ** int(shot['side_of_basket'] == "R"))
            distance = int(math.sqrt((50 * base_pct_adjusted) ** 2 + (side_pct * 94) ** 2))

            # Create shot instance
            newShot = self._PlayerShots(
                player_id=shot['player'],
                team_id=shot['team'],
                opp_id=self.teamIds["home"].split(".")[-1] if int(self.teamIds["home"].split(".")[-1]) != int(shot["team"]) else self.teamIds["away"].split(".")[-1],
                game_id=self.gameId.split(".")[-1],
                period=shot["period"],
                shot_type_id=shot['type'],
                assist_id=shot['assister'],
                shot_made=shot['shot_made'],
                points=int(shot['points']),
                base_pct=base_pct,
                side_pct=side_pct,
                distance=distance,
                fastbreak=shot['fastbreak'],
                side_of_basket=shot['side_of_basket'],
                clutch=self._calculate_clutch(shot["clock"], int(shot["period"])/2),
                zone=self._get_shot_zone(shot['side_of_basket'], side_pct, base_pct)
            )
            playerShots.append(newShot)
        boxscore.player_shots = playerShots
        

    def _set_misc(self, boxscore: Any, data: Dict[str, Any]) -> None:
        """Process a list of records and return a BoxScore with player shots."""
        self._set_player_shots(boxscore, data)
            
                
