from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Date, ForeignKey, CheckConstraint, text
from sqlalchemy.orm import relationship

from .database import Base 


class OverUnder(Base):
    __tablename__ = 'over_unders'
    game_id = Column(String, ForeignKey('games.game_id'), primary_key=True)
    over_under = Column(Float, nullable=False)
    over_line = Column(Integer, default=-110)
    under_line = Column(Integer, default=-110)
    total = Column(Integer, nullable=False)
    ou_outcome = Column(Integer, nullable=False)

    game = relationship("Game")

