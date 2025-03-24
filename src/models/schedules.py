from datetime import datetime, timedelta
from typing import List

from ..capabilities import Processable

####################################################################
####################################################################



class Schedule(Processable):

    @staticmethod
    def is_active(schedule: dict) -> bool:
         # Convert string dates to datetime.date objects
        startDate = datetime.strptime(schedule.get("start_date"), "%Y-%m-%d")
        endDate = datetime.strptime(schedule.get("end_date"), "%Y-%m-%d")

        # Correct logic: today must be on or after start_date and before end_date
        return startDate <= datetime.today() < endDate
    

    def process(schedule: dict, nGD: int=0) -> List[str]:
        """
            Used to process boxscore lists and matchup lists by use of nGD or number of GameDates
                -  nGD > 0 adds today plus n-1 games of matchup GameDates
                -  an empty dateString means last_update = None 
        """
        raise NotImplementedError
    


####################################################################
####################################################################



class DailySchedule(Schedule):
        
    
    @staticmethod
    def process(schedule: dict, nGD: int=0) -> List[str]:

        gameDateList = []
        if schedule.get("last_update"):
            startDate = datetime.strptime(schedule.get("last_update"), "%Y-%m-%d").date()+timedelta(1)
        else:
            startDate = datetime.strptime(schedule.get("start_date"), "%Y-%m-%d").date()

        tempDate = datetime.strptime(schedule.get("end_date"), "%Y-%m-%d").date()
        endDate = tempDate if tempDate < datetime.today().date() else datetime.today().date() + timedelta(nGD)

        gameDateList = []
        while startDate < endDate:
            gameDateList.append(str(startDate))
            startDate += timedelta(1)
        print(gameDateList)
        return gameDateList



####################################################################
####################################################################



class WeeklySchedule(Schedule):
    
    @staticmethod
    def process(schedule: dict, nGD: int=0) -> List[str]:
        raise NotImplementedError
    
        
        