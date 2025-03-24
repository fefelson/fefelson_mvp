from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Date, ForeignKey, CheckConstraint, text
from sqlalchemy.orm import relationship

from .database import Base 

class Player(Base):
    __tablename__ = 'players'
    player_id = Column(String, primary_key=True)
    sport_id = Column(String, ForeignKey('sports.sport_id', ondelete='CASCADE'), nullable=False)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    height = Column(Integer, nullable=True)
    weight = Column(Integer, nullable=True)
    bats = Column(String, nullable=True)
    throws = Column(String, nullable=True)
    position = Column(String, nullable=True)
    birthdate = Column(Date, nullable=True)
    current_team_id = Column(String, ForeignKey('teams.team_id', ondelete='SET NULL'), nullable=False)
    uniform_number = Column(String, nullable=True)
    draft_year = Column(Integer, nullable=True)
    draft_pick = Column(Integer, nullable=True)
    draft_team_id = Column(String, ForeignKey('teams.team_id', ondelete='SET NULL'), nullable=True)
    graduation_yr = Column(String, nullable=True)

    sport = relationship("Sport", back_populates="players")
    draft_team = relationship("Team", foreign_keys=[draft_team_id], back_populates="draft_players")
