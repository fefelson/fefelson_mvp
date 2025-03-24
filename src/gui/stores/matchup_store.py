from datetime import datetime
import json
import os
import pytz

from ...models.model_classes import MatchupData

est = pytz.timezone('America/New_York')

class MatchupStore:

    def __init__(self):

        self.matchups = {}
        for leagueId in ("NCAAB", "NBA"):
            today = datetime.now().astimezone(est)
            month, day = str(today.date()).split("-")[1:]
            module_dir = os.path.dirname(os.path.abspath(__file__))
            basePath = f"{module_dir}/../../../data/matchups/{leagueId}/2024/{month}/{day}/"

            for filePath in [basePath+fileName for fileName in os.listdir(basePath)]:
                with open(filePath) as fileIn:
                    matchup = MatchupData(**json.load(fileIn))
                    self.matchups[matchup.gameId] = matchup
            

    def get_gameDate(self):
        return [m for m in sorted(self.matchups.values(), key=lambda x: datetime.fromisoformat(x.gameTime)) if datetime.fromisoformat(m.gameTime) > datetime.now().astimezone(est) ]
    

    def get_matchup(self, gameId):
        return self.matchups[gameId]
    

    def get_tracking(self, identifier, gameId):
        matchup = self.matchups[gameId]
        if identifier == "pts_spread":
            timestamps = [record['timestamp'] for record in matchup.odds if record['home_spread'] != ""]
            lineItem = [abs(float(record['home_spread'])) for record in matchup.odds if record['home_spread'] != ""]  # Convert to int for plotting

            # Parse timestamps into datetime objects
            datetimes = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f%z') for ts in timestamps]

            # Calculate time differences in seconds relative to the first timestamp
            x_seconds = [(dt - datetimes[0]).total_seconds() / 60 for dt in datetimes]  # Convert to minutes for better scaling

            # For x-axis labels, use the time part of the datetime
            time_labels = [dt.strftime('%a %H:%M') for dt in datetimes]

        return {
            "xData":x_seconds,
            "yData":lineItem,
            "xLabels":time_labels,
            "label": lineItem[-1]
        }

            

        