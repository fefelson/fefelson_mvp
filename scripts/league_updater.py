#!/usr/bin/env python3

from typing import Optional

import sys

from src.sports.basketball.leagues import NBA, NCAAB


leagues = {"NBA": NBA, "NCAAB": NCAAB}

def main(leagueId: Optional[str] = None) -> None:
    """Main function to update leagues based on input."""
    if leagueId:
        league = leagues[leagueId]()
        league.update()
    else:
        for league in leagues.values():
            league().update()
         

if __name__ == "__main__":
    # Parse command-line arguments
    league_id = sys.argv[1] if len(sys.argv) > 1 else None

   
    main(league_id)
    # except Exception as e:
    #     sys.exit(1)  # Exit with error code for cron to detect failure
    # else:
    #     sys.exit(0)  # Exit successfully