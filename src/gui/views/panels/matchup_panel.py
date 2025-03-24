import wx


class MatchupPanel(wx.ScrolledWindow):

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, style=wx.VSCROLL | wx.HSCROLL, *args, **kwargs)

        self.SetScrollbars(20, 20, 10, 10)
        self.sizer = wx.BoxSizer(wx.VERTICAL)

        