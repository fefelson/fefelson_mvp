from datetime import datetime 
import os
import wx




class ThumbPanel(wx.Panel):
    def __init__(self, parent, ctrl=None, size=wx.Size(170, 180), style=wx.BORDER_SIMPLE, *args, **kwargs):
        wx.Panel.__init__(self, parent, size=size, style=style, *args, **kwargs)
        self.SetMaxSize(wx.Size(170, 180))

        # Font definitions
        gdFont = wx.Font(7, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False, wx.EmptyString)
        gtFont = wx.Font(9, wx.FONTFAMILY_ROMAN, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False, wx.EmptyString)
        moneyFont = wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False, wx.EmptyString)
        lineFont = wx.Font(8, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, False, wx.EmptyString)
        teamFont = wx.Font(11, wx.FONTFAMILY_MODERN, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False, wx.EmptyString)

        # Game info
        self.gameDate = wx.StaticText(self, label="gamedate", size=wx.Size(90, 15), style=wx.ALIGN_CENTER_HORIZONTAL)
        self.gameDate.SetFont(gdFont)
        
        self.gameTime = wx.StaticText(self, label="gametime", size=wx.Size(90, 15), style=wx.ALIGN_CENTER_HORIZONTAL)
        self.gameTime.SetFont(gtFont)

        # Spread and Over/Under
        self.ptsSpread = wx.StaticText(self, label="-pts-", size=wx.Size(45, 15), style=wx.ALIGN_CENTER_HORIZONTAL)
        self.ptsSpread.SetFont(moneyFont)
        self.overUnder = wx.StaticText(self, label="-o/u-", size=wx.Size(45, 15), style=wx.ALIGN_CENTER_HORIZONTAL)
        self.overUnder.SetFont(moneyFont)

        # Team data containers
        self.logos = {}
        self.abrvs = {}
        self.moneyLines = {}
        self.spreadLines = {}
        self.overLines = {}

        # Initialize team widgets
        for a_h in ("away", "home"):
            self.logos[a_h] = wx.StaticBitmap(self, wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.Size(45, 45), 0)
            self.abrvs[a_h] = wx.StaticText(self, wx.ID_ANY, label=a_h.upper(), size=wx.Size(45, 20), style=wx.ALIGN_CENTER_HORIZONTAL | wx.ST_ELLIPSIZE_END)
            self.abrvs[a_h].SetFont(teamFont)
            self.moneyLines[a_h] = wx.StaticText(self, wx.ID_ANY, label="-110", size=wx.Size(45, 15), style=wx.ALIGN_CENTER_HORIZONTAL)
            self.moneyLines[a_h].SetFont(moneyFont)
            self.spreadLines[a_h] = wx.StaticText(self, wx.ID_ANY, label="-110", size=wx.Size(45, 15), style=wx.ALIGN_CENTER_HORIZONTAL)
            self.spreadLines[a_h].SetFont(lineFont)

        for o_u in ("over", "under"):
            self.overLines[o_u] = wx.StaticText(self, label="-110", size=wx.Size(45, 15), style=wx.ALIGN_CENTER_HORIZONTAL)
            self.overLines[o_u].SetFont(lineFont)

        for comp in (self.spreadLines["home"], self.spreadLines["away"], self.overLines["under"], self.overLines["over"]):
            comp.Hide()

        # GridBagSizer layout
        grid_sizer = wx.GridBagSizer(hgap=5, vgap=2)  # Reduced vgap for tighter spacing

        # Row 0-1: Game info (spans all columns)
        grid_sizer.Add(self.gameDate, pos=(0, 0), span=(1, 3), flag=wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, border=5)
        grid_sizer.Add(self.gameTime, pos=(1, 0), span=(1, 3), flag=wx.ALIGN_CENTER_HORIZONTAL, border=5)

        # Row 2-7: Team sections and middle column
        # Away team (Column 0)
        grid_sizer.Add(self.abrvs["away"], pos=(2, 0), flag=wx.ALIGN_CENTER_HORIZONTAL, border=5)
        grid_sizer.Add(self.logos["away"], pos=(3, 0), flag=wx.ALIGN_CENTER_HORIZONTAL | wx.ALL, border=5)
        grid_sizer.Add(self.moneyLines["away"], pos=(4, 0), flag=wx.ALIGN_CENTER_HORIZONTAL, border=5)
        grid_sizer.Add(self.spreadLines["away"], pos=(5, 0), flag=wx.ALIGN_CENTER_HORIZONTAL, border=5)
        grid_sizer.Add(self.overLines["under"], pos=(6, 0), flag=wx.ALIGN_CENTER_HORIZONTAL, border=5)

        # Middle column (Column 1) - Spread and Over/Under
        grid_sizer.Add((0, 20), pos=(2, 1))  # Spacer to align with abrvs
        # grid_sizer.Add((0, 45), pos=(3, 1))  # Spacer to align with logos
        grid_sizer.Add((0, 20), pos=(4, 1))  # Spacer to align with money lines
        grid_sizer.Add(self.ptsSpread, pos=(5, 1), flag=wx.ALIGN_CENTER_HORIZONTAL | wx.EXPAND, border=5)
        grid_sizer.Add(self.overUnder, pos=(6, 1), flag=wx.ALIGN_CENTER_HORIZONTAL | wx.EXPAND, border=5)

        # Home team (Column 2)
        grid_sizer.Add(self.abrvs["home"], pos=(2, 2), flag=wx.ALIGN_CENTER_HORIZONTAL, border=5)
        grid_sizer.Add(self.logos["home"], pos=(3, 2), flag=wx.ALIGN_CENTER_HORIZONTAL | wx.ALL, border=5)
        grid_sizer.Add(self.moneyLines["home"], pos=(4, 2), flag=wx.ALIGN_CENTER_HORIZONTAL, border=5)
        grid_sizer.Add(self.spreadLines["home"], pos=(5, 2), flag=wx.ALIGN_CENTER_HORIZONTAL, border=5)
        grid_sizer.Add(self.overLines["over"], pos=(6, 2), flag=wx.ALIGN_CENTER_HORIZONTAL, border=5)

        if ctrl:
            self.Bind(wx.EVT_LEFT_DCLICK, ctrl.on_click)
            self.ptsSpread.Bind(wx.EVT_LEFT_DCLICK, ctrl.on_spread)
            self.overUnder.Bind(wx.EVT_LEFT_DCLICK, ctrl.on_total)
            [self.moneyLines[a_h].Bind(wx.EVT_LEFT_DCLICK, ctrl.on_money) for a_h in ("away", "home")]
            
            for item in (*self.abrvs.values(), *self.logos.values()):
                item.Bind(wx.EVT_LEFT_DCLICK, ctrl.on_team)

            for item in (self.gameDate, self.gameTime):
                item.Bind(wx.EVT_LEFT_DCLICK, ctrl.on_click)
                        
        
        # Set the sizer and layout
        self.SetSizer(grid_sizer)
        self.Fit()
        self.Layout()


    def setPanel(self, game):
        gameId = game.gameId
        self.SetName(f"{gameId}") # Name of panel is set to gameId for event handling
        self.gameDate.SetName(f"{gameId}")
        self.gameTime.SetName(f"{gameId}")


        gd = datetime.fromisoformat(game.gameTime)
        self.gameDate.SetLabel(gd.strftime("%a %b %d"))
        self.gameTime.SetLabel(gd.strftime("%I:%M %p"))

        try:
            for i in range(len(game.odds)):
                x = game.odds[(i+1)*-1].get("home_spread", None)
                y = game.odds[(i+1)*-1].get("awaySpread", None)
                if x:
                    if float(x) > 0:
                        spread = f"+{x}"
                    else:
                        spread = x
                    break
                elif y:
                    y = float(y)*-1
                    if y > 0:
                        spread = f"+{y}"
                    else:
                        spread = y
                    break
            self.ptsSpread.SetLabel(f"{spread}")
        except:
            spread = "n/a"
        self.ptsSpread.SetName(f"{spread} {gameId}")
        

        try:
            oU = None
            for i in range(len(game.odds)):
                x = game.odds[(i+1)*-1].get("total", None)
                if x:
                    oU = x
                    break
            self.overUnder.SetLabel("{:.1f}".format(float(oU)))
        except:
            oU = "n/a"
        self.overUnder.SetName(f"{oU} {gameId}")


        
        for a_h in ("away", "home"):
            index = a_h == "home"
            team = game.teams[index]
            teamId = team["team_id"]
            self.abrvs[a_h].SetName(f"{teamId} {gameId}")
            self.abrvs[a_h].SetLabel(team["abbr"])

            self.logos[a_h].SetName(f"{teamId} {gameId}")

            module_dir = os.path.dirname(os.path.abspath(__file__))
            logoPath = module_dir+"/../../../../data/{}_logos/{}.png".format(game.leagueId.lower(), team["team_id"].split(".")[-1])
            
            try:
                logo = wx.Image(logoPath, wx.BITMAP_TYPE_PNG).Scale(45, 45, wx.IMAGE_QUALITY_HIGH)
            except:
                logo = wx.Image(f"{module_dir}/../../../../data/ncaab_logos/-1.png", wx.BITMAP_TYPE_PNG).Scale(45, 45, wx.IMAGE_QUALITY_HIGH)

            logo = logo.ConvertToBitmap()
            self.logos[a_h].SetBitmap(logo)

            try:
                mL = None
                for i in range(len(game.odds)):

                    x = game.odds[(i+1)*-1].get(f"{a_h}_ml", None)
                    if x:
                        if int(x) > 0:
                            mL = f"+{x}"
                        else:
                            mL = x
                        break
                self.moneyLines[a_h].SetLabel(f"{mL}")
            except AssertionError:
                self.moneyLines[a_h].SetLabel("--")
            
            self.moneyLines[a_h].SetName(f"{a_h}_{mL} {gameId}")

        self.Layout()

class MatchupTickerPanel(wx.Panel):
    def __init__(self, parent, size=(185, -1), *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.scrolledPanel = wx.ScrolledWindow(self, size=(185, -1), style=wx.VSCROLL)
        self.scrolledPanel.SetScrollbars(20, 20, 10, 10)
        self.scrolledPanel.SetScrollRate(10, 10)
        self.scrollSizer = wx.BoxSizer(wx.VERTICAL)
        self.scrolledPanel.SetSizer(self.scrollSizer)
        self.sizer = wx.BoxSizer()
        self.sizer.Add(self.scrolledPanel, 0, wx.EXPAND | wx.LEFT, 10)
        self.SetSizer(self.sizer)

        self.scrolledPanel.Bind(wx.EVT_MOUSEWHEEL, self.onMouseWheel)

    def onMouseWheel(self, event):
        event.Skip()  # Enable smooth scrolling

if __name__ == "__main__":
    app = wx.App()
    frame = wx.Frame(None, size=(850, 190))
    frame.SetSize(wx.Size(850,190))
    sizer = wx.BoxSizer(wx.VERTICAL)
    panel = MatchupTickerPanel(frame)

    for i in range(7):
        thumb = ThumbPanel(panel.scrolledPanel)
        panel.scrollSizer.Add(thumb, 1, wx.TOP | wx.BOTTOM, 25)
        thumb.Show()
    
    
    sizer.Add(panel, 0, wx.LEFT, 10)
    frame.SetSizer(sizer)
    
    frame.FitInside()
    frame.Show()
    app.MainLoop()

