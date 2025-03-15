from dataclasses import dataclass
from typing import Any, Dict, List, Optional


#######################################################################
#######################################################################


@dataclass
class PlayerData:
    player_id: str
    first_name: str
    last_name: str
    uniform_number: Optional[str]
    position: Optional[str] = None  # From primary_position_id
    current_team_id: Optional[int] = None


@dataclass
class StadiumData:
    stadium_id: str  # Yahoo uses string IDs
    name: str
    location: Optional[str] = None


@dataclass
class TeamData:
    team_id: str
    league_id: str
    first_name: str
    last_name: str
    abbreviation: str
    conference: Optional[str] = None
    division: Optional[str] = None
    primary_color: Optional[str] = None
    secondary_color: Optional[str] = None


@dataclass
class GameData:
    provider: str
    game_id: str
    league_id: str
    home_team_id: str
    away_team_id: str
    game_date: str  
    season: int
    game_type: str  # "Regular Season" or "Playoffs"
    game_result: str
    url: Optional[str] = None
    odds: Optional[dict] = None
    week: Optional[int] = None
    winner_id: Optional[str] = None
    loser_id: Optional[str] = None
    stadium_id: Optional[str] = None
    is_neutral_site: bool = False
    

@dataclass
class GameLineData:
    team_id: str
    opp_id: str
    game_id: str
    spread: float
    money_line: Optional[int]  
    result: int
    spread_outcome: int
    money_outcome: Optional[int]
    spread_line: Optional[int] = -110 


@dataclass
class OverUnderData:
    game_id: str
    over_under: float
    total: int
    ou_outcome: int
    over_line: Optional[int] = -110
    under_line: Optional[int] = -110
    

@dataclass
class PeriodData:
    game_id: str
    team_id: str
    opp_id: str
    period: int
    pts: int
    

@dataclass
class LineupData:
    position: Optional[str]
    player_id: str
    starter: bool
    order: int


@dataclass
class PlayerStatData:
    player_id: str
    game_id: str
    team_id: str
    opp_id: str


@dataclass
class TeamStatData:
    team_id: str
    game_id: str
    opp_id: str


@dataclass
class BoxscoreData:
    game: GameData
    team_stats: List[TeamStatData]
    player_stats: List[PlayerStatData]
    periods: List[PeriodData]
    game_lines: List[GameLineData]
    over_unders: OverUnderData
    lineups: List[LineupData]
    teams: List[TeamData]
    players: List[PlayerData]
    stadiums: StadiumData




@dataclass
class MatchupData:
    provider: str
    game_id: int
    gameTime: str
    homeId: int
    awayId: int
    url: str
    season: int
    week: int
    day: int
    game_type: str
    odds: List[Dict[str, Any]]
    players: Dict[str, Dict]
    teams: Dict[str, Dict]
    injuries: Dict[str, Any]
    

@dataclass
class ScoreboardData:
    provider: str
    league_id: str
    game_date: str  
    games: List[GameData]
    team_meta: Optional[List[TeamData]]

