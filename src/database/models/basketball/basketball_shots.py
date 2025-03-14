from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Date, ForeignKey, CheckConstraint, text
from sqlalchemy.orm import relationship

from ..database import Base 


class BasketballShot(Base):
    __tablename__ = 'basketball_shots'
    player_shot_id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(String, ForeignKey('games.game_id', ondelete='CASCADE'), nullable=False)
    player_id = Column(String, ForeignKey('players.player_id', ondelete='CASCADE'), nullable=False)
    team_id = Column(String, ForeignKey('teams.team_id', ondelete='SET NULL'), nullable=False)
    opp_id = Column(String, ForeignKey('teams.team_id', ondelete='SET NULL'), nullable=False)
    period = Column(Integer, nullable=False)
    shot_type_id = Column(Integer, ForeignKey('basketball_shot_types.shot_type_id', ondelete='SET NULL'), nullable=False)
    assist_id = Column(String, ForeignKey('players.player_id', ondelete='SET NULL'), nullable=True)
    shot_made = Column(Boolean, nullable=False)
    points = Column(Integer, nullable=False)
    base_pct = Column(Float, nullable=False)
    side_pct = Column(Float, nullable=False)
    distance = Column(Integer, nullable=False)
    side_of_basket = Column(String, nullable=False)
    clutch = Column(String, nullable=True)
    zone = Column(String, nullable=True)

    game = relationship("Game")
    player = relationship("Player", foreign_keys=[player_id])
    team = relationship("Team", foreign_keys=[team_id])
    opponent = relationship("Team", foreign_keys=[opp_id])
    assist_player = relationship("Player", foreign_keys=[assist_id])
    shot_type = relationship("BasketballShotType", back_populates="shots")