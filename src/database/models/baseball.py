from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey
from sqlalchemy.orm import relationship

from .database import Base 



##########################################################################
##########################################################################


class PitchType(Base):
    __tablename__ = 'pitch_types'
    pitch_type_id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)

    pitches = relationship("Pitch", back_populates="pitchType")


##########################################################################
##########################################################################


class PitchResult(Base):
    __tablename__ = 'pitch_results'
    pitch_result_id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)

    pitches = relationship("Pitch", back_populates="pitchResult")


##########################################################################
##########################################################################


class AtBatType(Base):
    __tablename__ = 'at_bat_types'
    at_bat_type_id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)

    atBats = relationship("AtBatResult", back_populates="at_bat_type")


##########################################################################
##########################################################################


class Pitch(Base):
    __tablename__ = 'pitches'
    pitch_id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(String, ForeignKey('games.game_id', ondelete='CASCADE'), nullable=False)
    pitcher_id = Column(String, ForeignKey('players.player_id', ondelete='CASCADE'), nullable=False)
    batter_id = Column(String, ForeignKey('players.player_id', ondelete='CASCADE'), nullable=False)
    pitch_type_id = Column(Integer, ForeignKey('pitch_types.pitch_type_id', ondelete='CASCADE'), nullable=False)
    pitch_result_id = Column(Integer, ForeignKey('pitch_results.pitch_result_id', ondelete='CASCADE'), nullable=False)
    period = Column(Integer, nullable=False)
    sequence = Column(Integer, nullable=False)
    balls = Column(Integer, nullable=False)
    strikes = Column(Integer, nullable=False)
    vertical = Column(Integer, nullable=False)
    horizontal = Column(Integer, nullable=False)
    velocity = Column(Integer, nullable=False)

    game = relationship("Game")
    batter = relationship("Player", foreign_keys=[batter_id])
    pitcher = relationship("Player", foreign_keys=[pitcher_id])
    pitchType = relationship("PitchType", back_populates="pitches")
    pitchResult = relationship("pitchResult", back_populates="pitches")
