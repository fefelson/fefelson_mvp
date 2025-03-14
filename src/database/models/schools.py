from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Date, ForeignKey, CheckConstraint, text
from sqlalchemy.orm import relationship

from .database import Base 

class School(Base):
    __tablename__ = 'schools'
    school_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    location = Column(String, nullable=True)

    teams = relationship("Team", back_populates="school")
