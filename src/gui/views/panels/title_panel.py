import os
import wx

from .base_panel import BasePanel
# from .tags_panel import TagPanel
# from ..helpers import adjust_readability, hex_to_rgb


imagePath = "/home/ededub/FEFelson/{}/logos/{}.png"
module_dir = os.path.dirname(os.path.abspath(__file__))

class TitlePanel(BasePanel):

    def __init__(self, parent, ctrl=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        labelFont = wx.Font(10, wx.FONTFAMILY_ROMAN, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, False, wx.EmptyString)
        nameFont = wx.Font(15, wx.FONTFAMILY_ROMAN, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False, wx.EmptyString)
        overallFont = wx.Font(25, wx.FONTFAMILY_ROMAN, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False, wx.EmptyString)
        boldLabel = wx.Font(12, wx.FONTFAMILY_ROMAN, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False, wx.EmptyString)


        self.logo = wx.StaticBitmap(self)
        self.logo.SetBitmap(wx.Image(f"{module_dir}/../../../../data/default.jpg", wx.BITMAP_TYPE_JPEG).Scale(100, 100, wx.IMAGE_QUALITY_HIGH).ConvertToBitmap())


        ###############################################
        self.firstName = wx.StaticText(self, label="First Name", style=wx.ST_ELLIPSIZE_END | wx.ALIGN_CENTER_HORIZONTAL)
        self.firstName.SetFont(nameFont)
        
        self.lastName = wx.StaticText(self, label="Last Name", style=wx.ST_ELLIPSIZE_END | wx.ALIGN_CENTER_HORIZONTAL)
        self.lastName.SetFont(nameFont)

        gpLabel = wx.StaticText(self, label="gp")
        gpLabel.SetFont(nameFont)
        self.gamesPlayed = wx.StaticText(self, label="0")
        self.gamesPlayed.SetFont(overallFont)

        gpSizer = wx.BoxSizer()
        gpSizer.Add(gpLabel, 0, wx.LEFT | wx.RIGHT, 15)
        gpSizer.Add(self.gamesPlayed)

        nameSizer = wx.BoxSizer(wx.VERTICAL)
        nameSizer.Add(self.firstName, 0, wx.EXPAND)
        nameSizer.Add(self.lastName, 0, wx.EXPAND)
        nameSizer.Add(gpSizer, 0, wx.TOP, 20)
        ###############################################



        ####### Overall Panel #########################
        ###############################################
        overallPanel = wx.Panel(self, style=wx.BORDER_SIMPLE)
        overallPanel.SetBackgroundColour(wx.WHITE)
        overallSizer = wx.BoxSizer(wx.VERTICAL)
        overallPanel.SetSizer(overallSizer)

        overallLabel = wx.StaticText(overallPanel, label="OVERALL")
        sosLabel = wx.StaticText(overallPanel, label="SOS")
        self.overallScore = wx.StaticText(overallPanel, label="00")
        self.sosScore = wx.StaticText(overallPanel, label="000")

        overallLabel.SetFont(labelFont)
        sosLabel.SetFont(labelFont)
        self.overallScore.SetFont(overallFont)
        self.sosScore.SetFont(nameFont)

        overallSizer.Add(overallLabel, 0, wx.ALIGN_CENTER_HORIZONTAL)
        overallSizer.Add(self.overallScore, 0, wx.ALIGN_CENTER_HORIZONTAL)
        overallSizer.Add(sosLabel, 0, wx.ALIGN_CENTER_HORIZONTAL)
        overallSizer.Add(self.sosScore, 0, wx.ALIGN_CENTER_HORIZONTAL)
        ###############################################
        ###############################################


        ##############################################
        offEffLabel = wx.StaticText(self, label= "Off", style=wx.ALIGN_CENTER_HORIZONTAL)
        defEffLabel = wx.StaticText(self, label="Def", style=wx.ALIGN_CENTER_HORIZONTAL)
        winPctLabel = wx.StaticText(self, label= "Win Pct:", style=wx.ALIGN_CENTER_HORIZONTAL)
        winRoiLabel = wx.StaticText(self, label= "ROI:", style=wx.ALIGN_CENTER_HORIZONTAL)
        atsPctLabel = wx.StaticText(self, label= "ATS Pct:", style=wx.ALIGN_CENTER_HORIZONTAL)
        atsRoiLabel = wx.StaticText(self, label= "ROI:", style=wx.ALIGN_CENTER_HORIZONTAL)
        self.ouPctLabel = wx.StaticText(self, label= "O/U PCT:", style=wx.ALIGN_CENTER_HORIZONTAL)
        ouRoiLabel = wx.StaticText(self, label= "ROI:", style=wx.ALIGN_CENTER_HORIZONTAL) 

        offEffLabel.SetFont(labelFont)
        defEffLabel.SetFont(labelFont)
        winPctLabel.SetFont(labelFont)
        winRoiLabel.SetFont(labelFont)
        atsPctLabel.SetFont(labelFont)
        atsRoiLabel.SetFont(labelFont)
        self.ouPctLabel.SetFont(labelFont)
        ouRoiLabel.SetFont(labelFont)      
       
        
        self.offEff = wx.StaticText(self, label = "000", style=wx.ALIGN_CENTER_HORIZONTAL)
        self.defEff = wx.StaticText(self, label= "000", style=wx.ALIGN_CENTER_HORIZONTAL)
        self.pace = wx.StaticText(self, label="Pace: Standard", style=wx.ALIGN_CENTER_HORIZONTAL)
        self.winPct = wx.StaticText(self, label = "00%", style=wx.ALIGN_CENTER_HORIZONTAL)
        self.winRoi = wx.StaticText(self, label= "0.00%", style=wx.ALIGN_CENTER_HORIZONTAL)
        self.atsPct = wx.StaticText(self, label="00%", style=wx.ALIGN_CENTER_HORIZONTAL)
        self.atsRoi = wx.StaticText(self, label = "0.00%", style=wx.ALIGN_CENTER_HORIZONTAL)
        self.ouPct = wx.StaticText(self, label= "00%", style=wx.ALIGN_CENTER_HORIZONTAL)
        self.ouRoi = wx.StaticText(self, label="0.00%", style=wx.ALIGN_CENTER_HORIZONTAL)

        self.offEff.SetFont(boldLabel)
        self.defEff.SetFont(boldLabel)
        self.pace.SetFont(boldLabel)
        self.winPct.SetFont(boldLabel)
        self.winRoi.SetFont(boldLabel)
        self.atsPct.SetFont(boldLabel)
        self.atsRoi.SetFont(boldLabel)
        self.ouPct.SetFont(boldLabel)
        self.ouRoi.SetFont(boldLabel)


        
        
        offSizer = wx.BoxSizer(wx.VERTICAL)
        offSizer.Add(offEffLabel, 0, wx.TOP | wx.BOTTOM, 10)
        offSizer.Add(self.offEff, 0)

        defSizer = wx.BoxSizer(wx.VERTICAL)
        defSizer.Add(defEffLabel, 0, wx.TOP | wx.BOTTOM, 10)
        defSizer.Add(self.defEff, 0)

        effSizer = wx.BoxSizer()
        effSizer.Add(offSizer, 0, wx.LEFT | wx.RIGHT, 10)
        effSizer.Add(defSizer, 0)


        gamingSizer = wx.GridBagSizer()
        gamingSizer.Add(winPctLabel, pos=(0,0), span=(1,2))
        gamingSizer.Add(self.winPct, pos=(0,3))
        gamingSizer.Add(winRoiLabel, pos=(0,6))
        gamingSizer.Add(self.winRoi, pos=(0,8))
        gamingSizer.Add(atsPctLabel, pos=(1,0), span=(1,2))
        gamingSizer.Add(self.atsPct, pos=(1,3))
        gamingSizer.Add(atsRoiLabel, pos=(1,6))
        gamingSizer.Add(self.atsRoi, pos=(1,8))
        gamingSizer.Add(self.ouPctLabel, pos=(3,0), span=(1,2))
        gamingSizer.Add(self.ouPct, pos=(3,3))
        gamingSizer.Add(ouRoiLabel, pos=(3,6))
        gamingSizer.Add(self.ouRoi, pos=(3,8))

        secondLevelSizer = wx.BoxSizer()
        secondLevelSizer.Add(effSizer, 1, wx.EXPAND)
        secondLevelSizer.Add(gamingSizer, 3, wx.EXPAND | wx.LEFT, 15)



        #############################################

        
        topSizer = wx.BoxSizer()
        topSizer.Add(overallPanel, 1, wx.EXPAND)
        topSizer.Add(nameSizer, 2)
        topSizer.Add(self.logo, 0)
              
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(topSizer, 0, wx.EXPAND)
        sizer.Add(secondLevelSizer, 0, wx.EXPAND)
        sizer.Add(self.pace, 0, wx.TOP | wx.ALIGN_CENTER_HORIZONTAL, 10)
        self.SetSizer(sizer)



