from ...database.models.database import get_db_session
from ...database.models.teams import Team 



class TeamStore:

    def __init__(self):
        pass 


    def get_team(self, teamId):
         with get_db_session() as session:
            team = session.query(Team).filter(
                Team.team_id == teamId).first()
            # Convert ORM object to dict and then to DataFrame
            return {column.name: getattr(team, column.name) for column in team.__table__.columns}


        