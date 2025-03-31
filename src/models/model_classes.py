from dataclasses import dataclass
from typing import Any, Dict, List, Optional


#######################################################################
#######################################################################


@dataclass
class BoxscoreData:
    game: Any
    teamStats: List[Any]
    playerStats: List[Any]
    periods: List[Any]
    gameLines: Optional[List[Any]]
    overUnders: Optional[List[Any]]
    lineups: Optional[List[Any]]
    teams: List[Any]
    players: List[Any]
    stadium: Any
    misc: Optional[List[Any]]




@dataclass
class MatchupData:
    provider: str
    gameId: str
    leagueId: str
    homeId: str
    awayId: str
    url: str
    gameTime: str
    season: int
    week: Optional[int]
    statusType: str
    gameType: str
    odds: List[Dict]
    lineups: Optional[Dict[str, Dict]] = None
    players: Optional[Dict] = None
    teams: Optional[List[Dict]] = None
    injuries: Optional[Dict] = None
    stadiumId: Optional[str] = None
    isNuetral: Optional[bool] = False
    

@dataclass
class ScoreboardData:
    provider: str
    league_id: str
    games: List[Any]

