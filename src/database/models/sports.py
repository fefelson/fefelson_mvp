from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Date, ForeignKey, CheckConstraint, text
from sqlalchemy.orm import relationship

from .database import Base 


# Core Tables
class Sport(Base):
    __tablename__ = 'sports'
    sport_id = Column(String, primary_key=True)  # e.g., "Basketball"
    name = Column(String, unique=True, nullable=False)

    leagues = relationship("League", back_populates="sport")
    players = relationship("Player", back_populates="sport")


