from abc import ABC, abstractmethod


################################################################################
################################################################################


class Updateable(ABC):


    @abstractmethod
    def needs_update(self):
        pass


    @abstractmethod
    def update(self):
        pass