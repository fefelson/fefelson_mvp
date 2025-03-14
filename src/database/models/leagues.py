from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Date, ForeignKey, CheckConstraint, text
from sqlalchemy.orm import relationship

from .database import Base 

class League(Base):
    __tablename__ = 'leagues'
    league_id = Column(String, primary_key=True)  # e.g., "NBA"
    sport_id = Column(String, ForeignKey('sports.sport_id'), nullable=False)
    name = Column(String, unique=True, nullable=False)

    sport = relationship("Sport", back_populates="leagues")
    teams = relationship("Team", back_populates="league")
    games = relationship("Game", back_populates="league")
