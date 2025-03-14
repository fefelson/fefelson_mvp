from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Date, ForeignKey, CheckConstraint, text
from sqlalchemy.orm import relationship

from .database import Base 

class Stadium(Base):
    __tablename__ = 'stadiums'
    stadium_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    location = Column(String, nullable=True)

    games = relationship("Game", back_populates="stadium")
