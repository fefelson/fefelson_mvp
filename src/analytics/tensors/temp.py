import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from pprint import pprint

from ..src.database.models.database import get_db_session

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

class HitClassifier(nn.Module):
    def __init__(self, num_hardness=4, num_style=3, num_stadium=35, embed_dim=6, hidden_dim=16, num_classes=9):
        super(HitClassifier, self).__init__()
        # Embedding layers for categorical inputs
        self.hardness_embed = nn.Embedding(num_hardness, 2)  # e.g., 4 unique hardness values
        self.style_embed = nn.Embedding(num_style, 2)        # e.g., 3 unique style values
        self.stadium_embed = nn.Embedding(num_stadium, 3)        # e.g., 3 unique style values
        
        # Input dim: 2 embeddings * embed_dim + 2 continuous = 2*2 + 2 = 6
        self.fc1 = nn.Linear(9, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, hardness_embed, style_embed, stadium_embed, cont_inputs):
        # cat_inputs: [batch_size, 2] (hardness, style)
        # cont_inputs: [batch_size, 2] (distance, angle)
        
        # Embed categorical inputs
        hardness = self.hardness_embed(hardness_embed)  # [batch_size, embed_dim]
        style = self.style_embed(style_embed)       # [batch_size, embed_dim]
        stadium = self.stadium_embed(stadium_embed)       # [batch_size, embed_dim]
     
        
        # Concatenate all inputs
        x = torch.cat([hardness, style, stadium, cont_inputs], dim=1)  # [batch_size, embed_dim*2 + 2]
        
        # Forward pass
        # x = self.fc1(x)  # [batch_size, hidden_dim]
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        # x = self.dropout(x)
        x = self.fc2(x)              # [batch_size, num_classes]

        return x
# F.silu(self.fc1(x))
# # Replace torch.relu with alternatives
# x = torch.nn.functional.leaky_relu(self.fc1(x), negative_slope=0.01)  # Leaky ReLU
# # or
# x = torch.nn.functional.elu(self.fc1(x), alpha=1.0)                   # ELU
# # or
# x = torch.nn.functional.gelu(self.fc1(x))     

def query_db():

    with get_db_session() as session:
        query = f"""
                SELECT hit_hardness, hit_style, hit_distance, hit_angle, stadiums.name, at_bats.at_bat_type_id
                FROM at_bats
                INNER JOIN at_bat_types ON at_bats.at_bat_type_id = at_bat_types.at_bat_type_id
                INNER JOIN games ON at_bats.game_id = games.game_id
                INNER JOIN stadiums ON games.stadium_id = stadiums.stadium_id
                WHERE at_bats.at_bat_type_id NOT IN (0,6)
                """
        return pd.read_sql(query, session.bind)  


def main():
    categorical_cols = ["hit_hardness", "hit_style"]
    continuous_cols = ["hit_distance", "hit_angle"]
    label_col = "at_bat_type_id"

    df = query_db()
    df = df.dropna()

    mapping = dict(zip(df["name"].unique(), range(len(df["name"].unique()))))
    df["name"] = df["name"].map(mapping)  # This is the key change


    scaler = StandardScaler()
    df[continuous_cols] = scaler.fit_transform(df[continuous_cols])

    trainFrame, testFrame = train_test_split(df, test_size=0.2)

    # Convert to tensors
    hard_train = torch.tensor(trainFrame["hit_hardness"].values, dtype=torch.long)
    style_train= torch.tensor(trainFrame["hit_style"].values, dtype=torch.long)
    stadium_train = torch.tensor(trainFrame["name"].values, dtype=torch.long)
    cont_train = torch.tensor(trainFrame[continuous_cols].values, dtype=torch.float32)
    labels = torch.tensor(trainFrame[label_col].values, dtype=torch.long)

    hard_test = torch.tensor(testFrame["hit_hardness"].values, dtype=torch.long)
    style_testn= torch.tensor(testFrame["hit_style"].values, dtype=torch.long)
    stadium_test = torch.tensor(testFrame["name"].values, dtype=torch.long)
    cont_test = torch.tensor(testFrame[continuous_cols].values, dtype=torch.float32)
    test_labels = torch.tensor(testFrame[label_col].values, dtype=torch.long)



    # Create dataset and dataloader
    dataset = TensorDataset(hard_train, style_train, stadium_train, cont_train, labels)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Instantiate model, loss, and optimizer
    model = HitClassifier(num_hardness=4, num_style=6, embed_dim=2, hidden_dim=64, num_classes=11)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    num_epochs = 50
    for epoch in range(num_epochs):
        total_loss = 0
        for hard_batch, style_batch, stadium_batch, cont_batch, label_batch in dataloader:
            optimizer.zero_grad()               # Clear gradients
            outputs = model(hard_batch, style_batch, stadium_batch, cont_batch)  # Forward pass
            loss = criterion(outputs, label_batch)  # Compute loss
            loss.backward()                     # Backpropagation
            optimizer.step()                    # Update weights
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    model.eval()
    with torch.no_grad():
        outputs = model(hard_test, style_testn, stadium_test, cont_test)
        _, predicted = torch.max(outputs, dim=1)
        accuracy = (predicted == test_labels).float().mean()
        print(f"Accuracy: {accuracy:.4f}")

    
    




def tensor_test(season=2024):
    # Load and preprocess data
    torch.manual_seed(42)
     # Enforce deterministic behavior
    torch.use_deterministic_algorithms(True)


    df = query_db(season)
    df = df.dropna()

    # Create deterministic team ID mapping
    all_teams = sorted(set(df["home_id"]))  # Sort for consistency
    team_to_idx = {team: idx for idx, team in enumerate(all_teams)}
    num_teams = len(team_to_idx)

    all_players = sorted(set(df['home_pitcher_id']).union(set(df['away_pitcher_id'])))  # Sort for consistency
    player_to_idx = {player: idx for idx, player in enumerate(all_players)}
    num_players = len(player_to_idx)

    # Map team IDs
    df["home_id"] = df["home_id"].map(team_to_idx)
    df["away_id"] = df["away_id"].map(team_to_idx)

    df["home_pitcher_id"] = df["home_pitcher_id"].map(player_to_idx)
    df["away_pitcher_id"] = df["away_pitcher_id"].map(player_to_idx)

    df['home_age_years'] = (df['home_age_years'].dt.days / 365.25).round(2)
    df["home_age_years"] = StandardScaler().fit_transform(df[["home_age_years"]])
    df["home_service_time"] = StandardScaler().fit_transform(df[["home_service_time"]])

    df['away_age_years'] = (df['away_age_years'].dt.days / 365.25).round(2)
    df["away_age_years"] = StandardScaler().fit_transform(df[["away_age_years"]])
    df["away_service_time"] = StandardScaler().fit_transform(df[["away_service_time"]])

    df['is_winner'] = df['is_winner'].astype(bool)


    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2)
    
    # Convert to tensors
    train_tensors = [torch.tensor(train_df[col].values, dtype=torch.long) 
                    for col in ['home_id', 'away_id', "home_pitcher_id", "away_pitcher_id"]]
    [train_tensors.append(torch.tensor(train_df[col].values, dtype=torch.float))
        for col in ["home_age_years", "home_service_time", "away_age_years", "away_service_time"]]
    train_tensors.append(torch.tensor(train_df['is_winner'].values, dtype=torch.float))
    
    test_tensors = [torch.tensor(test_df[col].values, dtype=torch.long) 
                    for col in ['home_id', 'away_id', "home_pitcher_id", "away_pitcher_id"]]
    [test_tensors.append(torch.tensor(test_df[col].values, dtype=torch.float))
        for col in ["home_age_years", "home_service_time", "away_age_years", "away_service_time"]]
    test_tensors.append(torch.tensor(test_df['is_winner'].values, dtype=torch.float))
    
    # Create datasets and loaders
    train_dataset = TensorDataset(*train_tensors)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = TensorDataset(*test_tensors)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # tensors = [torch.tensor(df[col].values, dtype=torch.long)
    #             for col in ["team_id", "opp_id"]]
    # tensors.append(torch.tensor(df["is_winner"].values, dtype=torch.float))

    # # Create datasets and loaders
    # train_dataset = TensorDataset(*tensors)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Model
    model = ResultPredictor()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(TRAIN_ITTERATIONS):  # Reduced for simplicity
        total_loss = 0
        for inputs, target in train_loader:
            optimizer.zero_grad()
            outputs = model(*inputs)
            loss = criterion(outputs, target.view(-1, 1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch+10}, Train BCE: {total_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        total_loss = 0
        correct = 0
        total = 0
        for home_id, away_id, home_pitcher_id, away_pitcher_id, home_age_years, home_service_time, away_age_years, away_service_time, target in test_loader:
            outputs = model(home_id, away_id, home_pitcher_id, away_pitcher_id, home_age_years, home_service_time, away_age_years, away_service_time)
            loss = criterion(outputs, target.view(-1, 1))
            total_loss += loss.item()
            # Accuracy
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            correct += (preds == target.view(-1, 1)).sum().item()
            total += target.size(0)
        print(f"SEASON: {season}  Test BCE: {total_loss/len(test_loader):.4f}, Accuracy: {correct/total:.4f}\n\n\n")

    # Save model
    torch.save(model.state_dict(), "/home/ededub/FEFelson/fefelson_mvp/data/torch_model.pth")

query = f"""
                WITH StartingPitchers AS (
                SELECT b.game_id, b.team_id, b.player_id, b.pitch_order
                FROM baseball_bullpen b
                WHERE b.pitch_order = 1
                )
                SELECT hp_team.team_id AS home_id, hp.player_id AS home_pitcher_id, 
                        AGE(g.game_date, hp.birthdate) AS home_age_years, EXTRACT(YEAR FROM g.game_date) - hp.rookie_season AS home_service_time,
                        ap_team.team_id AS away_id, ap.player_id AS away_pitcher_id, 
                        AGE(g.game_date, ap.birthdate) AS away_age_years, EXTRACT(YEAR FROM g.game_date) - ap.rookie_season AS away_service_time,
                        hp_team.team_id = g.winner_id AS is_winner
                FROM games g
                LEFT JOIN StartingPitchers AS hp_team ON g.game_id = hp_team.game_id AND g.home_id = hp_team.team_id
                LEFT JOIN players AS hp ON hp_team.player_id = hp.player_id
                LEFT JOIN StartingPitchers AS ap_team ON g.game_id = ap_team.game_id AND g.away_id = ap_team.team_id
                LEFT JOIN players AS ap ON ap_team.player_id = ap.player_id
                WHERE season != {season}
                ORDER BY g.game_id;
                """




if __name__ == "__main__":
    main()