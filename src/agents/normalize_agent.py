from abc import ABC, abstractmethod


######################################################################
######################################################################


class INormalAgent(ABC):
    
    @abstractmethod
    def normalize_scoreboard(self, webData: dict) -> "ScoreboardData":
        pass


    @abstractmethod
    def normalize_boxscore(self, webData: dict) -> "BoxscoreData":
        pass


