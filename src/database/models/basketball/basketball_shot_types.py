from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Date, ForeignKey, CheckConstraint, text
from sqlalchemy.orm import relationship

from ..database import Base 



# Basketball-Specific Tables
class BasketballShotType(Base):
    __tablename__ = 'basketball_shot_types'
    shot_type_id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)

    shots = relationship("BasketballShot", back_populates="shot_type")
