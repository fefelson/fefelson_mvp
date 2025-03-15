from dataclasses import dataclass
from typing import Optional, List

from ...models.model_classes import (BoxscoreData, TeamStatData, 
                                     PlayerStatData)


@dataclass
class BasketballTeamStatData(TeamStatData):
    minutes: int
    fga: int
    fgm: int
    fta: int
    ftm: int
    tpa: int
    tpm: int
    pts: int
    oreb: int
    dreb: int
    ast: int
    stl: int
    blk: int
    turnovers: int
    fouls: int
    pts_in_pt: Optional[int] = None 
    fb_pts: Optional[int] = None 
    
    

@dataclass
class BasketballPlayerStatData(PlayerStatData):
    starter: bool
    mins: int
    fga: int
    fgm: int
    fta: int
    ftm: int
    tpa: int
    tpm: int
    pts: int
    oreb: int
    dreb: int
    ast: int
    stl: int
    blk: int
    turnovers: int
    fouls: int
    plus_minus: int = None
    

@dataclass
class BasketballShotData(PlayerStatData):
    period: int
    shot_type_id: str
    assist_id: Optional[int]
    shot_made: bool
    points: int
    base_pct: float
    side_pct: float
    distance: int
    fastbreak: bool
    side_of_basket: str
    clutch: bool
    zone: str
    

@dataclass
class BasketballBoxscoreData(BoxscoreData):
    shot_chart: Optional[List[BasketballShotData]] = None

    