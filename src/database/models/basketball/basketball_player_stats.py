from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Date, ForeignKey, CheckConstraint, text
from sqlalchemy.orm import relationship

from ..database import Base 


class BasketballPlayerStat(Base):
    __tablename__ = 'basketball_player_stats'
    player_id = Column(String, ForeignKey('players.player_id', ondelete='CASCADE'), primary_key=True)
    game_id = Column(String, ForeignKey('games.game_id', ondelete='CASCADE'), primary_key=True)
    team_id = Column(String, ForeignKey('teams.team_id', ondelete='SET NULL'), nullable=False)
    opp_id = Column(String, ForeignKey('teams.team_id', ondelete='SET NULL'), nullable=False)
    starter = Column(Boolean, default=False, nullable=False)
    minutes = Column(Float, nullable=False)
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
    plus_minus = Column(Integer, nullable=True)

    player = relationship("Player")
    team = relationship("Team", foreign_keys=[team_id])
    game = relationship("Game")
    opponent = relationship("Team", foreign_keys=[opp_id])





