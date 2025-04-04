import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset



class ExpandableEmbedding(nn.Module):
    def __init__(self, discreteN, initial_dim=4):
        super().__init__()
        self.embedding_dim = initial_dim
        self.embeddings = nn.Embedding(discreteN, initial_dim)

    def expand(self, new_dim):
        """ Expands embedding layer without losing old information. """
        old_weights = self.embeddings.weight.data  # Store old weights
        discreteN, old_dim = old_weights.shape

        # Create new embedding layer
        new_embedding_layer = nn.Embedding(discreteN, new_dim)

        # Copy old values
        new_embedding_layer.weight.data[:, :old_dim] = old_weights

        # Initialize new dimensions randomly
        nn.init.xavier_uniform_(new_embedding_layer.weight.data[:, old_dim:])

        # Replace old embeddings with new one
        self.embeddings = new_embedding_layer
        self.embedding_dim = new_dim

    def forward(self, discreteType):
        return self.embeddings(discreteType)
    



class PitchDictDataset(Dataset):
    def __init__(self, xData, yData):

        for outcome in yData:
            if 
        if yData[-1] in (0, 1):
            self._xData = xData
            self._yData = yData
                        
            train_pitches = TensorDataset(
                torch.tensor([d[0] for d in xTrain], dtype=torch.long),
                torch.tensor([d[1] for d in xTrain], dtype=torch.float),
                torch.tensor([d[2] for d in xTrain], dtype=torch.float),
                torch.tensor([d[3] for d in xTrain], dtype=torch.float),
                torch.tensor([d[0] for d in yTrain], dtype=torch.bool),
                torch.tensor([d[1] for d in yTrain], dtype=torch.bool),
                torch.tensor([d[2] for d in yTrain], dtype=torch.long))
        self.X_data = X_data
        self.y_data = y_data

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        x = self.X_data[idx]
        y = self.y_data[idx]
        return {
            "pitch_type": torch.tensor(x[0], dtype=torch.long),
            "velocity": torch.tensor(x[1], dtype=torch.float),
            "horiz":    torch.tensor(x[2], dtype=torch.float),
            "vert":     torch.tensor(x[3], dtype=torch.float),
            "isBall": torch.tensor(y[0], dtype=torch.bool),
        }
    


class Trainer:
    def __init__(self, model, criterion, optimizer, target_field, metric_fn=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.target_field = target_field
        self.metric_fn = metric_fn or self.default_accuracy

        

    def train(self, train_loader, epochs=10):
        for epoch in range(epochs):
            self.model.train()
            total_loss, correct, total = 0, 0, 0

            

            for batch in train_loader:
                pitch_type, velocity, horiz, vert, isSwing, isStrike, outcome = batch
                target = eval(self.target_field)  # e.g., 'isSwing'

                self.optimizer.zero_grad()
                y_pred = self.model(pitch_type, velocity, horiz, vert)
                loss = self.criterion(y_pred.squeeze(), target.float())
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                correct += (y_pred.squeeze().round() == target).sum().item()
                total += target.size(0)

            acc = correct / total
            print(f"[{self.target_field}] Epoch {epoch}, Loss: {total_loss:.4f}, Acc: {acc:.4f}")

    def default_accuracy(self, preds, targets):
        return (preds.round() == targets).float().mean()