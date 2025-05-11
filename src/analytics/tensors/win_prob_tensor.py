import torch
import torch.nn as nn


class ResultPredictor(nn.Module):
    def __init__(self, num_teams=30, num_pitchers=901, team_embed_dim=2, pitch_embed_dim=0, hidden_dim=32):
        super().__init__()
        
        # Embedding layers
        self.home_team_embed = nn.Embedding(num_teams, team_embed_dim)
        # self.home_pitcher_embed = nn.Embedding(num_pitchers, pitch_embed_dim)
        
        self.away_team_embed = nn.Embedding(num_teams, team_embed_dim)
        # self.away_pitcher_embed = nn.Embedding(num_pitchers, pitch_embed_dim)

        # Fully connected layers
        # self.fc1 = nn.Linear((team_embed_dim*2)+(pitch_embed_dim*2)+4, hidden_dim)
        self.fc1 = nn.Linear(team_embed_dim*2+pitch_embed_dim*2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout()
        
    def forward(self, home_id, away_id, home_pitcher_id, away_pitcher_id, home_age_years, home_service_time, away_age_years, away_service_time):
        # Example: Get embeddings (2D tensors, e.g., [batch_size, embedding_dim])
        team_x = self.home_team_embed(home_id)          # [batch_size, embedding_dim]
        opp_x = self.away_team_embed(away_id)           # [batch_size, embedding_dim]
        # pitch_x = self.home_pitcher_embed(home_pitcher_id)  # [batch_size, embedding_dim]
        # opp_pitch = self.away_pitcher_embed(away_pitcher_id)  # [batch_size, embedding_dim]

        # Reshape scalar inputs to 2D (from [batch_size] to [batch_size, 1])
        home_age_years = home_age_years.unsqueeze(1)    # [batch_size, 1]
        home_service_time = home_service_time.unsqueeze(1)  # [batch_size, 1]
        away_age_years = away_age_years.unsqueeze(1)    # [batch_size, 1]
        away_service_time = away_service_time.unsqueeze(1)  # [batch_size, 1]

        # # Concatenate along feature dimension (dim=1)
        # x = torch.cat([
        #     team_x, pitch_x, 
        #     home_age_years, home_service_time,
        #     opp_x, opp_pitch,
        #     away_age_years, away_service_time
        # ], dim=1)  # [batch_size, total_features]        
        x = torch.cat([
            team_x, #pitch_x,
            opp_x, #opp_pitch
        ], dim=1)  # [batch_size, total_features] 
        
        x = torch.relu(self.fc1(x))
        x = self.fc2(x) 
        # x = self.dropout(x)
        return x  