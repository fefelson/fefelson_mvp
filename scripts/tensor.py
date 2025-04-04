import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split

from ..src.analytics.tensors.core import ExpandableEmbedding

# File Paths (Modify for DB interaction later)
DATA_FILE = "/home/ededub/FEFelson/mlb/data.json"
MODEL_CHECKPOINT = "pitch_model.pth"
EMBEDDINGS_FILE = "pitch_embeddings.pth"





class SwingModel(nn.Module):
    def __init__(self, embedding_layer, hidden_dim=32):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.fc1 = nn.Linear(embedding_layer.embedding_dim + 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Binary classification

    def forward(self, pitch_type, velocity, x, y):
        pitch_embed = self.embedding_layer(pitch_type)
        x = torch.cat([pitch_embed, velocity.unsqueeze(1), x.unsqueeze(1), y.unsqueeze(1)], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Binary classification (swing or no swing)
        return x
    

class StrikeModel(nn.Module):
    def __init__(self, embedding_layer, hidden_dim=32):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.fc1 = nn.Linear(embedding_layer.embedding_dim + 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Binary classification

    def forward(self, pitch_type, velocity, x, y):
        pitch_embed = self.embedding_layer(pitch_type)
        x = torch.cat([pitch_embed, velocity.unsqueeze(1), x.unsqueeze(1), y.unsqueeze(1)], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Binary classification (strike or ball)
        return x
    

    def setStrikeData(self, xTrain, yTrain):
        
            train_loader = DataLoader(train_pitches, batch_size=64, shuffle=True)
    


# Model Definition
class PitchOutcomeModel(nn.Module):
    def __init__(self, num_pitch_types, embedding_dim=4, hidden_dim=32):
        super().__init__()
        self.pitch_embedding = nn.Embedding(num_pitch_types, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim + 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 5)
        self.relu = nn.ReLU()

    def forward(self, pitch_type, velocity, x, y):
        pitch_embed = self.pitch_embedding(pitch_type)
        x = torch.cat([pitch_embed, velocity.unsqueeze(1), x.unsqueeze(1), y.unsqueeze(1)], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Save & Load Functions
def save_model(model, optimizer, filepath=MODEL_CHECKPOINT):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, filepath)

def load_model(model, optimizer, filepath=MODEL_CHECKPOINT):
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Model Loaded Successfully")
    else:
        print("No checkpoint found.")

def save_embeddings(model, filepath=EMBEDDINGS_FILE):
    torch.save(model.pitch_embedding.weight.detach(), filepath)

def load_embeddings(model, filepath=EMBEDDINGS_FILE):
    if os.path.exists(filepath):
        model.pitch_embedding.weight.data.copy_(torch.load(filepath))
        print("Embeddings Loaded Successfully")

# Data Preprocessing
def load_data(filepath=DATA_FILE):
    with open(filepath) as file:
        info = json.load(file)
    data = []
    for pitch in info["p"]:
        try:
            result = 4 if int(pitch["result"]) == 10 else int(pitch["result"])
            data.append((int(pitch["pitch_type"]), int(pitch["velocity"]), int(pitch["horizontal"]), int(pitch["vertical"]), result))
        except KeyError:
            pass
    return data

def normalize_data(data):
    horizontal_values = np.array([d[2] for d in data])
    vertical_values = np.array([d[3] for d in data])
    velocity_values = np.array([d[1] for d in data])
    
    h_mean, h_std = np.mean(horizontal_values), np.std(horizontal_values)
    v_mean, v_std = np.mean(vertical_values), np.std(vertical_values)
    vel_mean, vel_std = np.mean(velocity_values), np.std(velocity_values)
    
    def preprocess(pitch_type, velocity, horizontal, vertical, isSwing, isStrike, result):
        return (pitch_type, (velocity - vel_mean) / vel_std, (horizontal - h_mean) / h_std, (vertical - v_mean) / v_std, isSwing, isStrike, result)
    
    return [preprocess(*d) for d in data]

# Training Pipeline
def train_model(model, train_loader, optimizer, criterion, epochs=50):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        
        for batch in train_loader:
            pitch_type, velocity, horiz, vert, isSwing, isStrike, target = batch
            optimizer.zero_grad()
            y_pred = model(pitch_type, velocity, horiz, vert)
            loss = criterion(y_pred, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            correct += (y_pred.argmax(dim=1) == target).sum().item()
            total += target.size(0)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {running_loss / len(train_loader):.4f}, Accuracy: {correct / total:.4f}')
    save_model(model, optimizer)
    save_embeddings(model)

# Main Execution
if __name__ == "__main__":
    raw_data = load_data()
    processed_data = normalize_data(raw_data)
    X_train, X_test, y_train, y_test = train_test_split(
        [x[:-3] for x in processed_data], [x[-3:] for x in processed_data], test_size=0.2, random_state=42)
    
    
    
    num_pitch_types = 10

    # Initial embedding layer
    pitchTypeEmbeddings = ExpandableEmbedding(discreteN==10, initial_dim=4)

    # Train basic model (swing or no swing) using 4-dim embeddings
    strikeZoneModel = StrikeZoneModel(pitchTypeEmbeddings)
    train_model(swing_model, ...)

    # Later, expand embeddings for more complex task
    embedding_layer.expand(new_dim=8)

    # Now train the next-level model (e.g., ball vs. strike)
    strike_model = StrikeModel(embedding_layer)
    train_model(strike_model, ...)
    
    swingModel = SwingModel()
    
    
    strikeModel = StrikeModel()
    model = PitchOutcomeModel(num_pitch_types)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    load_model(model, optimizer)
    load_embeddings(model)
    train_model(model, train_loader, optimizer, criterion)
