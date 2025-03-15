from abc import ABC, abstractmethod
from sqlalchemy.orm import Session
from typing import Any, List, Optional


###################################################################
###################################################################



class IDatabaseAgent(ABC):
    """Abstract interface for database operations."""

    @abstractmethod
    def save(self, data: Any) -> None:
        """Saves the provided data to the database."""
        pass

    @abstractmethod
    def load(self, identifier: Any) -> Optional[Any]:
        """Loads data from the database based on an identifier (e.g., ID or key)."""
        pass

    @abstractmethod
    def query(self, criteria: dict) -> List[Any]:
        """Queries the database with given criteria and returns matching results."""
        pass


###################################################################
###################################################################



class SQLAlchemyDatabaseAgent(IDatabaseAgent):
    """SQLAlchemy implementation of the IDatabaseAgent interface."""

    def __init__(self, session: "Session", model_class: Any):
        """Initialize with an SQLAlchemy session and the model class to operate on."""
        self.session = session
        self.model_class = model_class  # e.g., a class inheriting from Base

    def save(self, data: Any) -> None:
        """Saves the provided data to the database."""
        if isinstance(data, dict):
            db_object = self.model_class(**data)
        else:
            db_object = data  # Assume it's already an instance of the model
        self.session.add(db_object)
        self.session.commit()

    def load(self, identifier: Any) -> Optional[Any]:
        """Loads data from the database based on an identifier (e.g., ID)."""
        result = self.session.query(self.model_class).filter_by(id=identifier).first()
        return result

    def query(self, criteria: dict) -> List[Any]:
        """Queries the database with given criteria and returns matching results."""
        query = self.session.query(self.model_class)
        for key, value in criteria.items():
            query = query.filter(getattr(self.model_class, key) == value)
        return query.all()