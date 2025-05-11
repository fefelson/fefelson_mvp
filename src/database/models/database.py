from contextlib import contextmanager
from dotenv import dotenv_values
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import declarative_base, sessionmaker

import os


envVals = dotenv_values()

db_url = f"postgresql://{envVals['DB_USER']}:{envVals['DB_PASSWORD']}@{envVals['DB_HOST']}:{envVals['DB_PORT']}/{envVals['DB_NAME']}"
if not all([envVals['DB_USER'], envVals['DB_PASSWORD'], envVals['DB_HOST'], envVals['DB_PORT'], envVals['DB_NAME']]):
    raise ValueError("Some database environment variables are missing.")
os.environ["DATABASE_URL"] = db_url



# Define Base
Base = declarative_base()
# Create engine with the URL from alembic.ini
engine = create_engine(db_url, echo=False)  # echo=True for debugging
# Create a session factory
SessionFactory = sessionmaker(bind=engine)
# Optional: Function to initialize the database (for manual table creation)
def init_db():
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_db_session():
    session = SessionFactory()
    try:
        yield session
        session.commit()  # Commits if no exceptions occur
    except IntegrityError as e:
        session.rollback()
        raise  # Re-raises the IntegrityError
    except Exception as e:
        session.rollback()
        raise  # Re-raises any other exception
    finally:
        session.close()  # Always closes the session