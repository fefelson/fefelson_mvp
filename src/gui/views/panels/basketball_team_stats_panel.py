import wx
import matplotlib
matplotlib.use('WxAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure


class BasketballTeamStatsPanel(wx.ScrolledWindow):


    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, style=wx.VSCROLL, *args, **kwargs)
        self.SetScrollbars(20, 20, 10, 10)


        valueFont = wx.Font(10, wx.FONTFAMILY_ROMAN, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False, wx.EmptyString)
        teamFont = wx.Font(15, wx.FONTFAMILY_ROMAN, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False, wx.EmptyString)
        oppFont =wx.Font(12, wx.FONTFAMILY_ROMAN, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, False, wx.EmptyString)


        self.teamLabel = wx.StaticText(self, label="Team", style=wx.ALIGN_CENTER_HORIZONTAL)
        self.teamLabel.SetFont(teamFont)
        
        self.oppLabel = wx.StaticText(self, label="Opp", style=wx.ALIGN_CENTER_HORIZONTAL)
        self.oppLabel.SetFont(oppFont)

        charts = {}
        for off_def in ("offense", "defense"):
            fig1 = Figure(figsize=(1.5, 0.2),  layout='constrained')
            ax1 = fig1.add_subplot(111)
            charts[off_def] = FigureCanvas(self, -1, fig1)
        
        
        nameSizer = wx.BoxSizer()
        nameSizer.Add(self.oppLabel, 1, wx.EXPAND)
        nameSizer.Add(wx.StaticLine(self, size=(-1,15)), 1, wx.EXPAND)
        nameSizer.Add(self.teamLabel, 1, wx.EXPAND)

        ptsSizer = wx.BoxSizer()
        ptsSizer.Add(charts["defense"], 1, wx.EXPAND)
        ptsSizer.Add(wx.StaticText(self, label="Points"), 1, wx.EXPAND)
        ptsSizer.Add(charts["offense"], 1, wx.EXPAND)


        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(nameSizer, 0, wx.EXPAND)
        sizer.Add(ptsSizer, 0, wx.EXPAND)
        self.SetSizer(sizer)        




    def set_panel(self, team):
        pass