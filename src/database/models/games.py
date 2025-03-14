from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Date, ForeignKey, CheckConstraint, text
from sqlalchemy.orm import relationship

from .database import Base 

class Game(Base):
    __tablename__ = 'games'
    game_id = Column(String, primary_key=True)
    league_id = Column(String, ForeignKey('leagues.league_id', ondelete='CASCADE'), nullable=False)
    home_team_id = Column(String, ForeignKey('teams.team_id', ondelete='SET NULL'), nullable=False)
    away_team_id = Column(String, ForeignKey('teams.team_id', ondelete='SET NULL'), nullable=False)
    winner_id = Column(String, ForeignKey('teams.team_id', ondelete='SET NULL'), nullable=True)
    loser_id = Column(String, ForeignKey('teams.team_id', ondelete='SET NULL'), nullable=True)
    stadium_id = Column(String, ForeignKey('stadiums.stadium_id'), nullable=True)
    is_neutral_site = Column(Boolean, default=False)
    game_date = Column(DateTime, nullable=False)
    season = Column(Integer, nullable=False)
    week = Column(Integer, nullable=True)
    game_type = Column(String, CheckConstraint("game_type IN ('season', 'postseason', 'final')"), nullable=False)
    game_result = Column(String, CheckConstraint("game_result IN ('won', 'tie')"), default='won')

    league = relationship("League", back_populates="games")
    home_team = relationship("Team", foreign_keys=[home_team_id], back_populates="home_games")
    away_team = relationship("Team", foreign_keys=[away_team_id], back_populates="away_games")
    winner = relationship("Team", foreign_keys=[winner_id], back_populates="won_games")
    loser = relationship("Team", foreign_keys=[loser_id], back_populates="lost_games")
    stadium = relationship("Stadium", back_populates="games")
