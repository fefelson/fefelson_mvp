from sqlalchemy import Column, Integer, String, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from .database import Base



class TeamProviderMapping(Base):
    __tablename__ = 'team_provider_mapping'
    id = Column(Integer, primary_key=True, autoincrement=True)
    team_id = Column(String, ForeignKey('teams.team_id'), nullable=False)
    provider_name = Column(String, nullable=False)  # e.g., "ProviderA"
    provider_team_id = Column(String, nullable=False)  # e.g., "nba_lakers"
    # team = relationship("Team", back_populates="provider_team_ids")
    __table_args__ = (UniqueConstraint('provider_name', 'provider_team_id'),)

class PlayerProviderMapping(Base):
    __tablename__ = 'player_provider_mapping'
    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(String, ForeignKey('players.player_id'), nullable=False)
    provider_name = Column(String, nullable=False)
    provider_player_id = Column(String, nullable=False)  # e.g., "nba_lebron_001"
    # player = relationship("Player", back_populates="provider_player_ids")
    __table_args__ = (UniqueConstraint('provider_name', 'provider_player_id'),)

class GameProviderMapping(Base):
    __tablename__ = 'game_provider_mapping'
    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(String, ForeignKey('games.game_id'), nullable=False)
    provider_name = Column(String, nullable=False)
    provider_game_id = Column(String, nullable=False)  # e.g., "nba_game_12345"
    # game = relationship("Game", back_populates="provider_game_ids")
    __table_args__ = (UniqueConstraint('provider_name', 'provider_game_id'),)

