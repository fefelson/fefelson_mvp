from src.database.models.games import Game
from src.database.models.game_lines import GameLine
from src.database.models.teams import Team
from src.database.models.database import get_db_session

from pprint import pprint




def list_all_games():
    with get_db_session() as session:
        games = session.query(Team).all()  # Returns list of Game objects
        return games

# Test it
result = list_all_games()
print(f"Found {len(result)} games:")
for game in result[:5]:  # First 5 for brevity
    print(game)
    