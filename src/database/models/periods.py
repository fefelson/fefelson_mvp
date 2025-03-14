from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Date, ForeignKey, CheckConstraint, text
from sqlalchemy.orm import relationship

from .database import Base 

class Period(Base):
    __tablename__ = 'periods'
    game_id = Column(String, ForeignKey('games.game_id'), primary_key=True)
    team_id = Column(String, ForeignKey('teams.team_id'), primary_key=True)
    opp_id = Column(String, ForeignKey('teams.team_id'), nullable=False)
    period = Column(Integer, primary_key=True)
    pts = Column(Integer, nullable=False)

    game = relationship("Game")
    team = relationship("Team", foreign_keys=[team_id])
    opponent = relationship("Team", foreign_keys=[opp_id])
