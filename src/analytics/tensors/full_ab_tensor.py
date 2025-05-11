import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the model
class PitchSelectModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PitchSelectModel, self).__init__()
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Separate heads for each output
        self.pitch_type = nn.Linear(hidden_dim, 17)  # Binary: sigmoid
        self.pitch_location = nn.Linear(hidden_dim, 400)  # Binary: sigmoid
        self.pitch_velocity = nn.Linear(hidden_dim, 1)  # Binary: sigmoid
        

    def forward(self, x):
        # Shared feature extraction
        features = self.shared(x)
        # Individual heads
        pitch_type = self.pitch_type(features)  
        pitch_location = self.pitch_location(features)  
        pitch_velocity = self.pitch_velocity(features) 
        return pitch_type, pitch_location, pitch_velocity
        

# Combined loss function
def combined_loss(outputs, targets, weights):
    """
    outputs: Tuple of (is_ob, is_hit, is_hr, is_k, num_bases) predictions
    targets: Tensor of shape [batch_size, 5] where:
             [:, 0:4] are binary labels (is_ob, is_hit, is_hr, is_k)
             [:, 4] is the num_bases label (0-4)
    weights: List of 5 weights for [is_ob, is_hit, is_hr, is_k, num_bases]
    """
    is_ob_pred, is_hit_pred, is_hr_pred, is_k_pred, num_bases_pred = outputs
    is_ob_true, is_hit_true, is_hr_true, is_k_true, num_bases_true = (
        targets[:, 0], targets[:, 1], targets[:, 2], targets[:, 3], targets[:, 4].long()
    )

    # Binary cross-entropy for binary outcomes
    bce_loss = nn.BCELoss()
    loss_is_ob = bce_loss(is_ob_pred.squeeze(), is_ob_true.float())
    loss_is_hit = bce_loss(is_hit_pred.squeeze(), is_hit_true.float())
    loss_is_hr = bce_loss(is_hr_pred.squeeze(), is_hr_true.float())
    loss_is_k = bce_loss(is_k_pred.squeeze(), is_k_true.float())

    # Categorical cross-entropy for num_bases
    cce_loss = nn.CrossEntropyLoss()
    loss_num_bases = cce_loss(num_bases_pred, num_bases_true)

    # Combine losses with weights
    total_loss = (
        weights[0] * loss_is_ob +
        weights[1] * loss_is_hit +
        weights[2] * loss_is_hr +
        weights[3] * loss_is_k +
        weights[4] * loss_num_bases
    )
    return total_loss

