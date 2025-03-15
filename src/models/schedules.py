from abc import abstractmethod
from datetime import datetime, timedelta
from typing import List

from ..capabilities import Fileable, Processable
from ..agents.file_agent import JSONAgent


####################################################################
####################################################################



class Schedule(Fileable, Processable):

    def __init__(self, leagueId: str):
        Fileable.__init__(self, JSONAgent())

        self.leagueId = leagueId
        self.set_file_path()
        self.config = self.read_file()


    @abstractmethod
    def _output_gamedate_list(self, gameDate: datetime) -> List[str]:
        raise NotImplementedError

    
    def set_file_path(self):
        self.filePath = f"data/{self.leagueId}_schedule.json"


    def is_active(self) -> bool:
         # Convert string dates to datetime.date objects
        startDate = datetime.strptime(self.config["start_date"], "%Y-%m-%d")
        endDate = datetime.strptime(self.config["end_date"], "%Y-%m-%d")

        # Correct logic: today must be on or after start_date and before end_date
        return startDate <= datetime.today() < endDate
    

    def process(self, dateString: str=None, nGD: int=0):
        """
            Used to process boxscore lists and matchup lists by use of nGD or number of GameDates
                -  nGD > 0 adds today plus n-1 games of matchup GameDates
                -  an empty dateString means last_update = None 
        """
        
        gameDateList = []
        if dateString:
            gameDate = datetime.strptime(dateString, "%Y-%m-%d").date()
            gameDateList = self._output_gamedate_list(gameDate, nGD)
        else:
            startDate = datetime.strptime(self.config["start_date"], "%Y-%m-%d").date()
            gameDateList = self._output_gamedate_list(startDate, nGD)
        return gameDateList



####################################################################
####################################################################



class DailySchedule(Schedule):

    def __init__(self, leagueId: str):
        super().__init__(leagueId)
        
        
    def _output_gamedate_list(self, gameDate: datetime, nGD: int=0) -> List[str]:
        gameDateList = []
        while gameDate < datetime.today().date()+timedelta(nGD):
            gameDateList.append(str(gameDate))
            gameDate += timedelta(1)
        return gameDateList



####################################################################
####################################################################



class WeeklySchedule(Schedule):
    
    def __init__(self, leagueId: str):
        super().__init__(leagueId)
        
        
    def _output_gamedate_list(self, gameDate: datetime, nGD: int=0) -> List[str]:
        raise NotImplementedError

