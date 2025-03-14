from sqlalchemy import Column, Computed, Integer, String, Float, Boolean, DateTime, Date, ForeignKey, CheckConstraint, text
from sqlalchemy.orm import relationship

from ..database import Base 


class BasketballTeamStat(Base):
    __tablename__ = 'basketball_team_stats'
    team_id = Column(String, ForeignKey('teams.team_id', ondelete='CASCADE'), primary_key=True)
    game_id = Column(String, ForeignKey('games.game_id', ondelete='CASCADE'), primary_key=True)
    opp_id = Column(String, ForeignKey('teams.team_id', ondelete='CASCADE'), nullable=False)
    minutes = Column(Integer, nullable=False)
    fga = Column(Integer, nullable=False)
    fgm = Column(Integer, nullable=False)
    fta = Column(Integer, nullable=False)
    ftm = Column(Integer, nullable=False)
    tpa = Column(Integer, nullable=False)
    tpm = Column(Integer, nullable=False)
    pts = Column(Integer, nullable=False)
    oreb = Column(Integer, nullable=False)
    dreb = Column(Integer, nullable=False)
    ast = Column(Integer, nullable=False)
    stl = Column(Integer, nullable=False)
    blk = Column(Integer, nullable=False)
    turnovers = Column(Integer, nullable=False)
    fouls = Column(Integer, nullable=False)
    pts_in_pt = Column(Integer, nullable=True)
    possessions = Column(Integer, Computed('fga + 0.44 * fta - oreb + turnovers'))

    team = relationship("Team", foreign_keys=[team_id])
    game = relationship("Game")
    opponent = relationship("Team", foreign_keys=[opp_id])
