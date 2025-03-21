from contextlib import contextmanager
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import declarative_base, sessionmaker

import os


load_dotenv()
# Construct the DATABASE_URL if it is not set
db_url = os.getenv("DATABASE_URL")
if not db_url:
    db_url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}" 
    if not all([os.getenv("DB_USER"), os.getenv("DB_PASSWORD"), os.getenv("DB_HOST"), os.getenv("DB_PORT"), os.getenv("DB_NAME")]):
        raise ValueError("Some database environment variables are missing.")
print(f"Database URL: {db_url}")  # Optional, to verify it works



# Define Base
Base = declarative_base()
# Create engine with the URL from alembic.ini
engine = create_engine(db_url, echo=True)  # echo=True for debugging
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