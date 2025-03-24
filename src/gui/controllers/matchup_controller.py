from datetime import datetime

from ..stores.game_line_store import GameLineStore
from ..stores.matchup_store import MatchupStore
from ..stores.team_store import TeamStore
from ..views.dashboards.matchup_dashboard import MatchupDashboard


###################################################################################
###################################################################################


class MatchupController:

    def __init__(self, frame):


        self.model = MatchupStore()
        self.teamStore = TeamStore()
        self.gameLine = GameLineStore()
        self.dashboard = MatchupDashboard(frame)     

        self.dashboard.new_thumb_panels(self.model.get_gameDate(), self) 
            
        

    def on_spread(self, event):
        obj = event.GetEventObject()
        ptsSpread, gameId = obj.GetName().split()


    def on_team(self, event):
        obj = event.GetEventObject()
        teamId, gameId = obj.GetName().split()

        team = self.teamStore.get_team(teamId)

        spreadTrack = self.model.get_tracking("pts_spread", gameId)
        self.dashboard.chartPanel.ptsSpreadTrack.set_panel(spreadTrack)

        gamingLog = self.gameLine.get_game_line_data(teamId)
        self.dashboard.chartPanel.ptsSpreadChart.set_panel(team, *self.gameLine.get_pts_spread_chart(gamingLog))
        self.dashboard.chartPanel.atsChart.set_panel(team, *self.gameLine.get_ats_chart(gamingLog))
        self.dashboard.chartPanel.winLossChart.set_panel(team, *self.gameLine.get_win_loss_chart(gamingLog))
        
        self.dashboard.chartPanel.Layout()




    def on_total(self, event):
        obj = event.GetEventObject()
        oU, gameId = obj.GetName().split()


    def on_money(self, event):
        obj = event.GetEventObject()
        moneyLine, gameId = obj.GetName().split()


    def on_click(self, event):
        obj = event.GetEventObject()
        gameId = obj.GetName()

        spreadTrack = self.model.get_tracking("pts_spread", gameId)
        self.dashboard.chartPanel.ptsSpreadTrack.set_panel(spreadTrack)
        self.dashboard.chartPanel.Layout()


        

        
        

        
        


    