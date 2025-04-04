import torch.nn as nn
import torch

class ExpandableEmbedding(nn.Module):
    def __init__(self, num_pitch_types, initial_dim=4):
        super().__init__()
        self.embedding_dim = initial_dim
        self.embeddings = nn.Embedding(num_pitch_types, initial_dim)

    def expand(self, new_dim):
        """ Expands embedding layer without losing old information. """
        old_weights = self.embeddings.weight.data  # Store old weights
        num_pitch_types, old_dim = old_weights.shape

        # Create new embedding layer
        new_embedding_layer = nn.Embedding(num_pitch_types, new_dim)

        # Copy old values
        new_embedding_layer.weight.data[:, :old_dim] = old_weights

        # Initialize new dimensions randomly
        nn.init.xavier_uniform_(new_embedding_layer.weight.data[:, old_dim:])

        # Replace old embeddings with new one
        self.embeddings = new_embedding_layer
        self.embedding_dim = new_dim

    def forward(self, pitch_type):
        return self.embeddings(pitch_type)
