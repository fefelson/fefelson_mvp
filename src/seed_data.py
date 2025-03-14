from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

import os
import sys
# Adjust sys.path to src/ from wherever the script runs
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


from database.models.database import engine, Base
from database.models.sports import Sport
from database.models.leagues import League
from database.models.basketball.basketball_shot_types import BasketballShotType

def seed_data():
    # Create tables if they don’t exist
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        try:
            # Seed Sports
            sports_data = [
                {"sport_id": "sport_basketball", "name": "Basketball"},
                {"sport_id": "sport_baseball", "name": "Baseball"},
                {"sport_id": "sport_football", "name": "Football"},
                {"sport_id": "sport_hockey", "name": "Hockey"}
            ]
            for data in sports_data:
                if not session.query(Sport).filter_by(sport_id=data["sport_id"]).first():
                    session.add(Sport(**data))

            # Seed Leagues
            leagues_data = [
                {"league_id": "league_nba", "sport_id": "sport_basketball", "name": "National Basketball Association"},
                {"league_id": "league_ncaab", "sport_id": "sport_basketball", "name": "College Basketball"},
                {"league_id": "league_mlb", "sport_id": "sport_baseball", "name": "Major League Baseball"},
                {"league_id": "league_nfl", "sport_id": "sport_football", "name": "National Football League"},
                {"league_id": "league_ncaaf", "sport_id": "sport_football", "name": "College Football"}
            ]
            for data in leagues_data:
                if not session.query(League).filter_by(league_id=data["league_id"]).first():
                    session.add(League(**data))

            # Seed Basketball Shot Types (ignoring shot_type field)
            shot_types_data = [
                {"shot_type_id": 1, "name": "General Field Goal Attempt (Mixed)"},  # Assuming 2, mixed
                {"shot_type_id": 2, "name": "Three-Point Shot Attempt"},
                {"shot_type_id": 3, "name": "Hook Shot Attempt"},
                {"shot_type_id": 5, "name": "Layup Attempt (Assisted)"},
                {"shot_type_id": 6, "name": "Driving Layup Attempt"},
                {"shot_type_id": 7, "name": "Dunk Attempt"},
                {"shot_type_id": 9, "name": "Driving Dunk Attempt"},
                {"shot_type_id": 10, "name": "Free Throw (Unspecified)"},
                {"shot_type_id": 11, "name": "First Free Throw"},
                {"shot_type_id": 12, "name": "Second Free Throw"},
                {"shot_type_id": 13, "name": "First Free Throw (Alternate)"},
                {"shot_type_id": 14, "name": "Second Free Throw (Alternate)"},
                {"shot_type_id": 15, "name": "Third Free Throw"},
                {"shot_type_id": 16, "name": "Technical Free Throw"},
                {"shot_type_id": 17, "name": "First Flagrant Foul Free Throw"},
                {"shot_type_id": 18, "name": "Second Flagrant Foul Free Throw"},
                {"shot_type_id": 19, "name": "Flagrant Foul Free Throw (Unspecified)"},
                {"shot_type_id": 21, "name": "First Technical Free Throw"},
                {"shot_type_id": 22, "name": "Second Technical Free Throw"},
                {"shot_type_id": 23, "name": "First Clear Path Free Throw"},
                {"shot_type_id": 24, "name": "Second Clear Path Free Throw"},
                {"shot_type_id": 25, "name": "Running Layup Attempt"},
                {"shot_type_id": 26, "name": "Alley-Oop Layup Attempt"},
                {"shot_type_id": 27, "name": "Reverse Layup Attempt"},
                {"shot_type_id": 28, "name": "Turnaround Jumper Attempt"},
                {"shot_type_id": 29, "name": "Running Dunk Attempt"},
                {"shot_type_id": 30, "name": "Reverse Dunk Attempt"},
                {"shot_type_id": 31, "name": "Alley-Oop Dunk Attempt"},
                {"shot_type_id": 34, "name": "Hook Shot Attempt (Generic)"},
                {"shot_type_id": 35, "name": "Turnaround Hook Shot Attempt"},
                {"shot_type_id": 40, "name": "Fade Away Jumper Attempt"},
                {"shot_type_id": 43, "name": "Miscellaneous Shot Attempt (Mixed)"},  # Assuming 2, mixed
                {"shot_type_id": 44, "name": "Bank Hook Shot Attempt"},
                {"shot_type_id": 45, "name": "Layup Attempt (General)"},
                {"shot_type_id": 46, "name": "Layup Attempt (Blocked)"},
                {"shot_type_id": 47, "name": "Layup Attempt (Mixed)"},
                {"shot_type_id": 48, "name": "Layup Attempt (Blocked)"},  # Duplicate name, distinct ID
                {"shot_type_id": 49, "name": "Layup Attempt (Unassisted)"},
                {"shot_type_id": 50, "name": "Layup Attempt (Mixed)"},  # Duplicate name, distinct ID
                {"shot_type_id": 52, "name": "Floating Jumper Attempt"},
                {"shot_type_id": 53, "name": "Pullup Jumper Attempt"},
                {"shot_type_id": 54, "name": "Step Back Jumper Attempt"},
                {"shot_type_id": 55, "name": "Pullup Bank Shot Attempt"},
                {"shot_type_id": 56, "name": "Driving Bank Shot Attempt"},
                {"shot_type_id": 57, "name": "Fade Away Bank Shot Attempt"},
                {"shot_type_id": 59, "name": "Turnaround Bank Shot Attempt"},
                {"shot_type_id": 60, "name": "Turnaround Fade Away Attempt"},
                {"shot_type_id": 61, "name": "Generic Dunk Attempt"},
                {"shot_type_id": 67, "name": "Driving Bank Hook Attempt"},
                {"shot_type_id": 70, "name": "Turnaround Bank Hook Attempt"},
                {"shot_type_id": 71, "name": "Tip Layup Attempt"},
                {"shot_type_id": 72, "name": "Cutting Layup Attempt"},
                {"shot_type_id": 73, "name": "Cutting Finger Roll Layup Attempt"},
                {"shot_type_id": 74, "name": "Running Alley-Oop Layup Attempt"},
                {"shot_type_id": 75, "name": "Driving Floating Jump Shot Attempt"},
                {"shot_type_id": 76, "name": "Driving Floating Bank Jump Shot Attempt"},
                {"shot_type_id": 79, "name": "Turnaround Fade Away Bank Jump Shot Attempt"},
                {"shot_type_id": 80, "name": "Running Alley-Oop Dunk Attempt"},
                {"shot_type_id": 81, "name": "Tip Dunk Attempt"},
                {"shot_type_id": 82, "name": "Cutting Dunk Attempt"},
                {"shot_type_id": 83, "name": "Driving Reverse Dunk Attempt"},
                {"shot_type_id": 84, "name": "Running Reverse Dunk Attempt"}
            ]
            for data in shot_types_data:
                if not session.query(BasketballShotType).filter_by(shot_type_id=data["shot_type_id"]).first():
                    session.add(BasketballShotType(**data))

            session.commit()
            print("Data seeded successfully!")
        except IntegrityError as e:
            session.rollback()
            print(f"Integrity error (likely duplicates skipped): {e}")
        except Exception as e:
            session.rollback()
            print(f"Seeding failed: {e}")
            raise

if __name__ == "__main__":
    seed_data()