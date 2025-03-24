import wx

from ..panels.chart_panel import ChartPanel
from ..panels.matchup_panel import MatchupPanel
from ..panels.matchup_ticker_panel import MatchupTickerPanel, ThumbPanel
from ..panels.team_panel import TeamPanel


emerald_fog = wx.Colour(50, 168, 82)         # #32A852

spacetime_funnel_cake = wx.Colour(245, 245, 220)  # #F5F5DC
# A soft, beige-cream, like funnel cake dusted with powdered sugar in a spacetime distortion.

rune_rose = wx.Colour(188, 143, 143)
dark_matter_haunted_house = wx.Colour(47, 79, 79)  # #2F4F4F
# A dark, slate teal, as mysterious as dark matter lurking in a haunted house.



class MatchupDashboard(wx.Panel):

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.SetBackgroundColour(dark_matter_haunted_house)

        self.sizer = wx.BoxSizer()

        self.tickerPanel = MatchupTickerPanel(self)
        self.tickerPanel.SetBackgroundColour(dark_matter_haunted_house)
        
        # self.matchupPanel = MatchupPanel(self)
        # self.matchupPanel.SetBackgroundColour(emerald_fog)

        self.teamPanel = TeamPanel(self)
        self.teamPanel.SetBackgroundColour(rune_rose)

        self.chartPanel = ChartPanel(self)
        self.chartPanel.SetBackgroundColour(dark_matter_haunted_house)

        self.sizer.Add(self.tickerPanel, 0, wx.EXPAND | wx.TOP, 10)
        # self.sizer.Add(self.matchupPanel, 13, wx.EXPAND)
        self.sizer.Add(self.teamPanel, 13, wx.EXPAND | wx.TOP, 10)
        self.sizer.Add(self.chartPanel, 7, wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT, 10)
        self.SetSizer(self.sizer)


    def new_thumb_panels(self, matchups, ctrl=None):
        for matchup in matchups:
            thumbPanel=ThumbPanel(self.tickerPanel.scrolledPanel, ctrl)
            self.tickerPanel.scrollSizer.Add(thumbPanel, 0, wx.BOTTOM, 10)
            thumbPanel.setPanel(matchup)
            thumbPanel.SetBackgroundColour(spacetime_funnel_cake)
            # thumbPanel.Show()

        


        

