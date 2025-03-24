import wx
from .gui.views.felson_frame import FelsonFrame
from .gui.controllers.matchup_controller import MatchupController
from .gui.views.splash_screen import SplashScreen

from .sports.basketball.leagues import NBA, NCAAB

class MainApp(wx.App):
    def OnInit(self):

        
        self.frame = FelsonFrame()
        splash = SplashScreen(self.frame)
        splash.Show()

        self.matchupController = MatchupController(self.frame)
        # self.compareTeamsController = CompareTeamsController(self.frame)

        
        # # Instantiate controllers with their respective panels
        # self.player_stats_controller = PlayerStatsController(self.frame.player_stats_panel)
        # self.game_log_controller = GameLogController(self.frame.game_log_panel)

        

        return True

if __name__ == "__main__":
    noLog = wx.LogNull()
    app = MainApp()
    app.MainLoop()