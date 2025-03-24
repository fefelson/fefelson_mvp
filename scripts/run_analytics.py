#!/usr/bin/env python3
from typing import Optional
import sys
import os
sys.path.append(os.path.expanduser('~/fefelson_mvp'))

from src.sports.basketball.leagues import NBA, NCAAB



leagues = {"NBA": NBA, "NCAAB": NCAAB}

def main(leagueId: Optional[str] = None) -> None:
    """Main function to update leagues based on input."""
    if leagueId:
        league = leagues[leagueId]()
        league.analyze()
    else:
        for league in leagues.values():
            league().analyze()
         

if __name__ == "__main__":
    # Parse command-line arguments
    league_id = sys.argv[1] if len(sys.argv) > 1 else None
    main(league_id)
