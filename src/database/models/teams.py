from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Date, ForeignKey, CheckConstraint, text
from sqlalchemy.orm import relationship

from .database import Base 

class Team(Base):
    __tablename__ = 'teams'
    team_id = Column(String, primary_key=True)
    league_id = Column(String, ForeignKey('leagues.league_id', ondelete='CASCADE'), nullable=False)
    school_id = Column(String, ForeignKey('schools.school_id'), nullable=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    abbreviation = Column(String, nullable=False)
    conference = Column(String, nullable=True)
    division = Column(String, nullable=True)
    primary_color = Column(String, nullable=True)
    secondary_color = Column(String, nullable=True)

    league = relationship("League", back_populates="teams")
    school = relationship("School", back_populates="teams")
    players = relationship("Player", foreign_keys="Player.current_team_id", back_populates="current_team")
    draft_players = relationship("Player", foreign_keys="Player.draft_team_id", back_populates="draft_team")
    home_games = relationship("Game", foreign_keys="Game.home_team_id", back_populates="home_team")
    away_games = relationship("Game", foreign_keys="Game.away_team_id", back_populates="away_team")
    won_games = relationship("Game", foreign_keys="Game.winner_id", back_populates="winner")
    lost_games = relationship("Game", foreign_keys="Game.loser_id", back_populates="loser")

