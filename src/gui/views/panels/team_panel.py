import wx

from .title_panel import TitlePanel
from .basketball_team_stats_panel import BasketballTeamStatsPanel

class TeamPanel(wx.Panel):

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.titlePanel = TitlePanel(self)
        
        self.gameLogPanel = wx.Panel(self)
        self.gameLogPanel.SetBackgroundColour(wx.Colour("blue"))

       
        self.noteBookPanel = wx.Notebook(self)
        self.teamStatsPanel = BasketballTeamStatsPanel(self.noteBookPanel)
        self.noteBookPanel.AddPage(self.teamStatsPanel, "Team Stats")

        self.playerStatsPanel = wx.Panel(self.noteBookPanel)
        self.noteBookPanel.AddPage(self.playerStatsPanel, "Player Stats")
       

        self.sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.leftSizer = wx.BoxSizer(wx.VERTICAL)
        self.leftSizer.Add(self.titlePanel, 3, wx.EXPAND)
        self.leftSizer.Add(self.noteBookPanel, 7, wx.EXPAND)

        self.sizer.Add(self.leftSizer, 4, wx.EXPAND)
        self.sizer.Add(self.gameLogPanel, 3, wx.EXPAND)

        self.SetSizer(self.sizer)

