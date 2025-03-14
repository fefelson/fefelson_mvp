from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from dotenv import load_dotenv
import configparser
import os

# Load .env from project root
load_dotenv()  # Looks for .env in current working directory (root)

# Get the database URL from the environment
db_url = os.getenv("DATABASE_URL")
if not db_url:
    raise ValueError("DATABASE_URL not set in environment")

# Define Base
Base = declarative_base()

# Create engine with the URL from alembic.ini
engine = create_engine(db_url, echo=True)  # echo=True for debugging

# Optional: Function to initialize the database (for manual table creation)
def init_db():
    Base.metadata.create_all(bind=engine)